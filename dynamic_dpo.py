import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
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
    
    # Set pad token to avoid attention mask issues
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

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

class CustomDPOTrainer(DPOTrainer):
    def log(self, logs, start_time=None):
        if start_time:
            logs["time"] = time.time() - start_time
        super().log(logs)

def train_dynamic_dpo(base_model, tokenizer, num_iterations=50):
    print("Starting Dynamic DPO Training...")
    
    def generate_prompt():
        inference_model = FastLanguageModel.for_inference(base_model)
        
        prompt_request = "Pretend you are a human using a language model and generate an interesting question or prompt that would test an AI's capabilities and knowledge. Output no other text but the prompt for the AI."
        messages = [{"from": "human", "value": prompt_request}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to("cuda")
        
        outputs = inference_model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=64,
            temperature=0.9,
            top_p=0.6,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
        
        prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prompt

    def generate_responses(prompt):
        inference_model = FastLanguageModel.for_inference(base_model)
        
        messages = [{"from": "human", "value": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to("cuda")
        
        responses = []
        configs = [
            {'temperature': 0.9, 'top_p': 0.6, 'top_k': 50, 'repetition_penalty': 1.2},
            {'temperature': 1.2, 'top_p': 0.4, 'top_k': 20, 'repetition_penalty': 1.5}
        ]
        
        for config in configs:
            outputs = inference_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **config
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        return responses

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
        
        attention_mask = torch.ones_like(inputs).to("cuda")
        
        outputs = inference_model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=200,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
        preference = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "response 1" in preference.lower():
            return response1, response2
        return response2, response1

    preference_data = []
    print(f"\nStarting dynamic preference collection for {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        prompt = generate_prompt()
        print(f"\nGenerated prompt: {prompt}")
        
        responses = generate_responses(prompt)
        print(f"\nResponse 1: {responses[0]}")
        print(f"\nResponse 2: {responses[1]}")
        
        chosen, rejected = get_preference(prompt, responses[0], responses[1])
        print(f"\nChosen response: {chosen}")
        
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
        if (iteration + 1) % 10 == 0:
            print(f"\nPerforming DPO update with {len(preference_data)} examples...")
            preference_dataset = Dataset.from_list(preference_data)
            
            training_args = DPOConfig(
                per_device_train_batch_size=2,
                learning_rate=5e-5,
                num_train_epochs=1,
                gradient_accumulation_steps=2,
                save_strategy="no",
                logging_steps=1,
                optim="adamw_8bit",
                remove_unused_columns=False,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                output_dir="dpo_output"
            )

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
            
            dpo_trainer.train()
            preference_data = []

    print("Dynamic DPO Training completed!")
    return base_model

def test_model(model, tokenizer, prompt="What is the meaning of life?"):
    print(f"\nTesting the model with prompt: {prompt}")
    model = FastLanguageModel.for_inference(model)
    test_messages = [
        {"from": "human", "value": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    attention_mask = torch.ones_like(inputs).to("cuda")
    
    print("Model response:")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        streamer=text_streamer,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )

if __name__ == "__main__":
    # First run SFT
    sft_model, tokenizer = train_sft()
    
    # Run dynamic DPO with 50 iterations
    final_model = train_dynamic_dpo(sft_model, tokenizer, num_iterations=50)
    
    # Test with multiple prompts
    test_prompts = [
        "What is the meaning of life?",
        "How does consciousness work?",
        "What is the future of technology?"
    ]
    
    for prompt in test_prompts:
        test_model(final_model, tokenizer, prompt)
