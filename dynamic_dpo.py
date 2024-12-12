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
    warnings.filterwarnings('ignore')
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        pad_token_id=50256,
        add_cross_attention=False,
        is_decoder=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        pad_token='<|endoftext|>',
        padding_side='left'
    )
    
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

def generate_prompt():
    prompts = [
        "What ethical considerations should guide AI development?",
        "How should AI handle conflicts between different ethical principles?",
        "What responsibilities do AI systems have towards human welfare?",
        "How should AI balance individual privacy with collective benefit?",
        "What role should AI play in decision-making about human lives?"
    ]
    import random
    return random.choice(prompts)

def generate_responses(model, tokenizer, prompt):
    inference_model = FastLanguageModel.for_inference(model)
    messages = [{"from": "human", "value": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    responses = []
    configs = [
        {'temperature': 0.7, 'top_p': 0.9, 'top_k': 50},
        {'temperature': 1.4, 'top_p': 0.3, 'top_k': 20}
    ]
    
    for config in configs:
        outputs = inference_model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs).to("cuda"),
            max_new_tokens=150,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,
            **config
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_response = clean_generated_text(generated_text)
        
        if cleaned_response.startswith(prompt):
            cleaned_response = cleaned_response[len(prompt):].strip()
        
        responses.append(cleaned_response)
    
    if responses[0] == responses[1]:
        return None
        
    return responses

def get_preference(prompt, response1, response2, rlhf_model, rlhf_tokenizer):
    if response1 is None or response2 is None:
        print("Skipping due to identical responses")
        return None, None
    
    # Simplified, more explicit prompt
    preference_prompt = f"""IMPORTANT: You must ONLY respond with two scores in exactly this format: "Score 1: X, Score 2: Y" where X and Y are numbers between 0 and 10.

Here are two responses to evaluate:

Response 1: {response1[:200]}...

Response 2: {response2[:200]}...

Consider:
- Ethical reasoning
- Factual accuracy
- Clarity and helpfulness

Respond ONLY with the scores of this format:
Score 1: 7, Score 2: 3
!!!! DO NOT RESPOND WITH THIS THESE NUMBERS. USE THE NUMBERS YOU DETERMINE TO BE REPRESENTATIVE OF THE QUALITY AND APPROPRIATENESS OF THE TWO RESPONSES YOU RECIEVED AND ACCORDING TO THE OUTLINED CONSIDERATIONS!!!!! 
"""
    
    inputs = rlhf_tokenizer(preference_prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    
    outputs = rlhf_model.generate(
        **inputs,
        max_new_tokens=10,  # Reduced to force shorter response
        num_return_sequences=1,
        temperature=0.1,    # Reduced to make output more consistent
        top_p=0.5,         # Reduced to make output more focused
        repetition_penalty=1.2,
        pad_token_id=rlhf_tokenizer.pad_token_id,
        do_sample=True,
        no_repeat_ngram_size=3
    )
    
    preference_text = rlhf_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"\nRaw RLHF output: {preference_text}")
    
    # Simplified regex pattern
    match = re.search(r"Score 1:\s*(\d+),\s*Score 2:\s*(\d+)", preference_text)
    if not match:
        print("Could not extract valid scores from RLHF output")
        return None, None
    
    try:
        score1 = float(match.group(1))
        score2 = float(match.group(2))
        
        # Ensure scores are between 0 and 10
        score1 = max(0, min(10, score1))
        score2 = max(0, min(10, score2))
        
        # Normalize to sum to 1.0
        total = score1 + score2
        if total > 0:
            score1 = score1 / total
            score2 = score2 / total
            print(f"Extracted scores: [{score1:.2f}], [{score2:.2f}]")
            return (response1, response2) if score1 > score2 else (response2, response1)
        else:
            print("Total score is 0")
            return None, None
    except ValueError:
        print("Invalid score values")
        return None, None
    
    try:
        score1 = float(match.group(1))
        score2 = float(match.group(2))
        
        total = score1 + score2
        if total > 0:
            score1 = score1 / total
            score2 = score2 / total
            print(f"Extracted scores: [{score1:.2f}], [{score2:.2f}]")
            return (response1, response2) if score1 > score2 else (response2, response1)
    except ValueError:
        print("Invalid score values extracted")
        return None, None
    
    return None, None

def train_dynamic_dpo(base_model, tokenizer, num_iterations):
    print("Starting Dynamic DPO Training...")
    
    rlhf_model, rlhf_tokenizer = prepare_rlhf_model()
    rlhf_model = rlhf_model.to("cuda")
    
    preference_data = []
    print(f"\nStarting dynamic preference collection for {num_iterations} iterations...")
    
    iteration = 0
    attempts = 0
    max_attempts = num_iterations * 3
    
    while iteration < num_iterations and attempts < max_attempts:
        attempts += 1
        print(f"\nIteration {iteration + 1}/{num_iterations} (Attempt {attempts})")
        
        prompt = generate_prompt()
        responses = generate_responses(base_model, tokenizer, prompt)
        if responses is None:
            print("Generated identical responses - retrying")
            continue
            
        chosen, rejected = get_preference(prompt, responses[0], responses[1], rlhf_model, rlhf_tokenizer)
        
        if chosen is None or rejected is None:
            print("Invalid preference pair - retrying")
            continue
        
        if chosen == rejected:
            print("Chosen and rejected responses are identical - retrying")
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
