import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, TextStreamer, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import time
import re
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

def clean_generated_text(text):
    """Remove template tokens and clean up the text."""
    # Remove template tokens
    cleaned = re.sub(r'<\|im_start\|>(?:user|assistant|system)', '', text)
    # Remove redundant newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned

def prepare_rlhf_model():
    """Prepare the RLHF model with proper tokenization and special tokens."""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        pad_token_id=50256,  # GPT2's EOS token ID
        add_cross_attention=False,
        is_decoder=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        pad_token='<|endoftext|>',
        padding_side='left'
    )
    
    # Add special tokens silently
    special_tokens = {
        'additional_special_tokens': ['*[', ']*', '[', ']'],
        'pad_token': '<|endoftext|>'
    }
    _ = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer

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

def get_preference(prompt, response1, response2, rlhf_model, rlhf_tokenizer):
    cleaned_prompt = clean_generated_text(prompt)
    cleaned_response1 = clean_generated_text(response1)
    cleaned_response2 = clean_generated_text(response2)
    
    preference_prompt = f"""Compare two responses and rate them:

Response 1: {cleaned_response1[:200]}...

Response 2: {cleaned_response2[:200]}...

Rate each response from 0 to 10. Higher is better.
Format your response exactly like this: Score 1: X, Score 2: Y
Example: Score 1: 7, Score 2: 4"""
    
    inputs = rlhf_tokenizer(preference_prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    
    outputs = rlhf_model.generate(
        **inputs,
        max_new_tokens=20,
        num_return_sequences=1,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=rlhf_tokenizer.pad_token_id,
        do_sample=True
    )
    
    preference_text = rlhf_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nRaw RLHF output: {preference_text}")
    
    scores = extract_preference_scores(preference_text)
    if scores is None:
        print("Could not extract valid preference scores - skipping this iteration")
        return None, None
        
    print(f"Extracted scores: [{scores[0]:.2f}], [{scores[1]:.2f}]")
    return (response1, response2) if scores[0] > scores[1] else (response2, response1)

def extract_preference_scores(text):
    patterns = [
        r"Score 1:\s*(\d+)[.,]?\s*Score 2:\s*(\d+)",
        r"(\d+)\s*[,/]\s*(\d+)",
        r"First.*?(\d+).*?Second.*?(\d+)",
        r"Response 1.*?(\d+).*?Response 2.*?(\d+)",
        r"(\d+).*?(\d+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                score1 = float(match.group(1))
                score2 = float(match.group(2))
                
                # Convert to 0-1 range if larger numbers were used
                if score1 > 1 or score2 > 1:
                    score1 = score1 / 10
                    score2 = score2 / 10
                
                # Normalize to sum to 1.0
                total = score1 + score2
                if total > 0:
                    score1 = score1 / total
                    score2 = score2 / total
                    return [score1, score2]
            except ValueError:
                continue
    
    return None  # Return None if no valid scores could be extracted

def train_dynamic_dpo(base_model, tokenizer, num_iterations):
    print("Starting Dynamic DPO Training...")
    
    rlhf_model, rlhf_tokenizer = prepare_rlhf_model()
    rlhf_model = rlhf_model.to("cuda")
    
    def generate_prompt():
        inference_model = FastLanguageModel.for_inference(base_model)
        prompt_request = "Generate an ethical dilemma or philosophical question that challenges AI decision-making."
        messages = [{"from": "human", "value": prompt_request}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        outputs = inference_model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs).to("cuda"),
            max_new_tokens=64,
            temperature=0.9,
            top_p=0.6,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_generated_text(generated_text)

    def generate_responses(prompt):
        inference_model = FastLanguageModel.for_inference(base_model)
        messages = [{"from": "human", "value": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        responses = []
        configs = [
            {'temperature': 0.6, 'top_p': 0.6},
            {'temperature': 1.2, 'top_p': 0.4}
        ]
        
        for config in configs:
            outputs = inference_model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs).to("cuda"),
                max_new_tokens=100,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                **config
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(clean_generated_text(generated_text))
        
        return responses

    preference_data = []
    print(f"\nStarting dynamic preference collection for {num_iterations} iterations...")
    
    iteration = 0
    while iteration < num_iterations:
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        prompt = generate_prompt()
        responses = generate_responses(prompt)
        chosen, rejected = get_preference(prompt, responses[0], responses[1], rlhf_model, rlhf_tokenizer)
        
        if chosen is None or rejected is None:
            print("Skipping iteration due to invalid scores")
            continue
        
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
        iteration += 1
        
        if len(preference_data) >= 10:
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

def test_model(model, tokenizer, prompts):
    responses = []
    for prompt in prompts:
        model = FastLanguageModel.for_inference(model)
        test_messages = [{"from": "human", "value": prompt}]
        inputs = tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs).to("cuda"),
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = clean_generated_text(tokenizer.decode(output[0], skip_special_tokens=True))
        responses.append(response)
    return responses

def generate_test_prompts():
    return [
        "What is the meaning of life?",
        "How does consciousness work?",
        "What is the future of technology?"
    ]

if __name__ == "__main__":
    test_prompts = generate_test_prompts()
    sft_model, tokenizer = train_sft()
    responses_before_dpo = test_model(sft_model, tokenizer, test_prompts)
    final_model = train_dynamic_dpo(sft_model, tokenizer, num_iterations=100)
    responses_after_dpo = test_model(final_model, tokenizer, test_prompts)

    with open("test_results.txt", "w") as file:
        file.write("Responses before DPO feedback loop:\n")
        for prompt, response in zip(test_prompts, responses_before_dpo):
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response: {response}\n\n")

        file.write("Responses after DPO feedback loop:\n")
        for prompt, response in zip(test_prompts, responses_after_dpo):
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Response: {response}\n\n")
