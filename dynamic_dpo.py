import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import time


def train_sft():
    print("Starting SFT Training...")
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )

    def apply_template(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
        return {"text": text}

    print("Loading SFT dataset...")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:100]")
    dataset = dataset.map(apply_template, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=5,
            output_dir="sft_output",
            seed=0,
        ),
    )

    print("Starting SFT training...")
    trainer.train()
    print("SFT Training completed!")
    return model, tokenizer

def train_dynamic_dpo(base_model, tokenizer, initial_prompts):
    print("Starting Dynamic DPO Training...")
    
    # Create a function to generate responses
    def generate_responses(prompt):
        inference_model = FastLanguageModel.for_inference(base_model)
    
        messages = [{"from": "human", "value": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
    
        attention_mask = torch.ones_like(inputs).to("cuda")
    
        responses = []
        # More diverse generation parameters
        configs = [
            {
                'temperature': 0.9,
                'top_p': 0.6,
                'top_k': 50,
                'repetition_penalty': 1.2
            },
            {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 30,
                'repetition_penalty': 1.3
            }
        ]
    
        for config in configs:
            outputs = inference_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                use_cache=True,
                **config
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
            # Add a small delay between generations to ensure different random seeds
            time.sleep(0.1)
    
        # Ensure responses are different
        if responses[0] == responses[1]:
            # Try again with even more different parameters
            config = {
                'temperature': 1.0,
                'top_p': 0.5,
                'top_k': 20,
                'repetition_penalty': 1.5
            }
            outputs = inference_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                use_cache=True,
                **config
            )
            responses[1] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        return responses

    # Function to get model's preference
    def get_preference(prompt, response1, response2):
        inference_model = FastLanguageModel.for_inference(base_model)
        comparison_prompt = f"""Compare these two responses and choose the better one:
        Prompt: {prompt}
        Response 1: {response1}
        Response 2: {response2}
        Explain your choice and clearly state which response (1 or 2) is better."""
        
        messages = [{"from": "human", "value": comparison_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = base_model.generate(
            input_ids=inputs,
            max_new_tokens=200,
            temperature=0.7,
            use_cache=True
        )
        preference = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Simple parsing of preference
        if "response 1" in preference.lower():
            return response1, response2
        return response2, response1

    # Collect dynamic preferences
    preference_data = []
    for prompt in initial_prompts:
        responses = generate_responses(prompt)
        chosen, rejected = get_preference(prompt, responses[0], responses[1])
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    # Create dataset from collected preferences
    from datasets import Dataset
    preference_dataset = Dataset.from_list(preference_data)

    # Use your existing DPO training setup
    training_args = DPOConfig(
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_steps=10,
        output_dir="dynamic_dpo_output",
        optim="adamw_8bit",
        remove_unused_columns=False,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported()
    )

    # Enable gradient checkpointing
    base_model.gradient_checkpointing_enable()

    class CustomDPOTrainer(DPOTrainer):
        def log(self, logs, start_time=None):
            if start_time:
                logs["time"] = time.time() - start_time
            super().log(logs)

    dpo_trainer = CustomDPOTrainer(
        model=base_model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=preference_dataset,
        args=training_args,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    print("Starting DPO training...")
    dpo_trainer.train()
    print("Dynamic DPO Training completed!")
    
    dpo_trainer.save_model("dynamic_final_model")
    return dpo_trainer.model

def test_model(model, tokenizer):
    print("\nTesting the model...")
    model = FastLanguageModel.for_inference(model)
    test_messages = [
        {"from": "human", "value": "What is the meaning of life?"},
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    print("Model response:")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        temperature=0.7,  # Add temperature for more interesting responses
        top_p=0.9,       # Add top_p for better sampling
        do_sample=True,  # Enable sampling
        use_cache=True
    )

# Modify your main function to use dynamic DPO
if __name__ == "__main__":
    # First run SFT
    sft_model, tokenizer = train_sft()
    
    # Define initial prompts for dynamic training
    initial_prompts = [
        "Explain quantum computing in simple terms",
        "What are the ethical implications of AI?",
        "How would you solve climate change?",
        "Describe the importance of space exploration"
    ]
    
    # Run dynamic DPO
    final_model = train_dynamic_dpo(sft_model, tokenizer, initial_prompts)
    
    # Test the final model
    test_model(final_model, tokenizer)
