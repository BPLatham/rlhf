import torch
from trl import SFTTrainer, DPOTrainer, DPOConfig, PPOTrainer, PPOConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, TrainingArguments, TextStreamer, BitsAndBytesConfig
from copy import deepcopy
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
import time

# First part: Supervised Fine-Tuning
def train_sft():
    print("Starting SFT Training...")
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return None, None

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    # Apply chat template before returning
    tokenizer = get_chat_template(
        tokenizer,
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        chat_template="chatml",
    )

    def apply_template(examples):
        try:
            # Process each example in the batch
            chosen_text = [f"Human: {chosen}\nAssistant: " for chosen in examples["chosen"]]
            rejected_text = [f"Human: {rejected}\nAssistant: " for rejected in examples["rejected"]]

            print("Sample chosen_text:", chosen_text[0] if chosen_text else "No chosen text")
            print("Sample rejected_text:", rejected_text[0] if rejected_text else "No rejected text")

            return {"chosen_text": chosen_text, "rejected_text": rejected_text}

        except Exception as e:
            print("Error applying template:", e)
            print("Failed example data:", examples)
            raise

    print("Loading SFT dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # Reduce dataset size for faster testing
    dataset = dataset.select(range(100))  # Select the first 100 examples

    # Log the first few entries to inspect structure
    print("Sample data from dataset:")
    try:
        print(dataset[0])
    except Exception as e:
        print("Error accessing dataset sample:", e)
        return None, None

    print("Applying template to dataset...")
    try:
        dataset = dataset.map(apply_template, batched=True)
    except Exception as e:
        print("Error while mapping dataset:", e)
        return None, None

    if dataset is None:
        print("Dataset mapping failed. Exiting.")
        return None, None

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="chosen_text",
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
    try:
        trainer.train()
    except Exception as e:
        print("Error during SFT training:", e)
        return None, None

    print("SFT Training completed!")
    return model, tokenizer



# Second part: PPO
from typing import Optional

def train_ppo_custom(base_model, tokenizer):
    print("Starting Custom PPO Training...")
    
    # Convert base_model for inference
    base_model = FastLanguageModel.for_inference(base_model)
    
    # Configuration
    class PPOConfigCustom:
        def __init__(self):
            self.learning_rate = 1e-5
            self.batch_size = 2
            self.epochs = 1
            self.steps_per_epoch = 100
            self.max_length = 2048
            self.kl_penalty = 0.1
            self.clip_epsilon = 0.2
            self.value_loss_coef = 0.1
            self.temperature = 0.7

    config = PPOConfigCustom()

    # Create value head
    class ValueHead(torch.nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.v_head = torch.nn.Linear(hidden_size, 1)
            self.v_head.to(base_model.device)

        def forward(self, hidden_states):
            return self.v_head(hidden_states)

    value_model = ValueHead()
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=config.learning_rate)

    # Create reward model
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "lvwerra/distilbert-imdb",
        num_labels=2,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16
    ).cuda()
    reward_model.eval()

    # Create dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    # Custom PPO step function
    def ppo_step(query, old_response, old_values, old_logprobs, rewards):
        # Forward pass
        outputs = base_model(
            query,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get new logprobs
        logits = outputs.logits
        new_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get new values
        hidden_states = outputs.hidden_states[-1]
        new_values = value_model(hidden_states)
        
        # Calculate advantages
        advantages = rewards - old_values
        
        # Calculate PPO policy loss
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Calculate value loss
        value_loss = torch.nn.functional.mse_loss(new_values, rewards + old_values)
        
        # Calculate KL penalty
        kl_div = (old_logprobs - new_logprobs).mean()
        
        # Total loss
        total_loss = policy_loss + config.value_loss_coef * value_loss + config.kl_penalty * kl_div
        
        return total_loss, policy_loss, value_loss, kl_div

    # Training loop
    try:
        base_model.train()
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch + 1}/{config.epochs}")
            
            for step in range(config.steps_per_epoch):
                batch = dataset[step * config.batch_size : (step + 1) * config.batch_size]
                
                # Prepare inputs
                inputs = tokenizer(
                    batch['chosen'], 
                    padding=True, 
                    truncation=True, 
                    max_length=config.max_length, 
                    return_tensors="pt"
                )
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
                
                # Generate initial responses
                with torch.no_grad():
                    outputs = base_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=config.batch_size,
                        do_sample=True,
                        temperature=config.temperature
                    )
                    initial_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Get rewards from reward model
                reward_inputs = tokenizer(batch['chosen'], initial_responses, 
                                       padding=True, 
                                       truncation=True, 
                                       max_length=config.max_length,
                                       return_tensors="pt")
                if 'token_type_ids' in reward_inputs:
                    del reward_inputs['token_type_ids']
                reward_inputs = {k: v.to(reward_model.device) for k, v in reward_inputs.items()}
                
                with torch.no_grad():
                    reward_outputs = reward_model(**reward_inputs)
                    rewards = reward_outputs.logits.softmax(dim=-1)[:, 1].detach()
                
                # Get old values and logprobs
                with torch.no_grad():
                    old_outputs = base_model(**inputs, output_hidden_states=True)
                    old_values = value_model(old_outputs.hidden_states[-1])
                    old_logprobs = torch.nn.functional.log_softmax(old_outputs.logits, dim=-1)
                
                # PPO update
                loss, policy_loss, value_loss, kl_div = ppo_step(
                    inputs,
                    initial_responses,
                    old_values,
                    old_logprobs,
                    rewards
                )
                
                # Optimization step
                optimizer.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                value_optimizer.step()
                
                if step % 10 == 0:
                    print(f"Step {step}: Loss: {loss.item():.4f}, "
                          f"Policy Loss: {policy_loss.item():.4f}, "
                          f"Value Loss: {value_loss.item():.4f}, "
                          f"KL Div: {kl_div.item():.4f}")

        # Convert model for inference before returning
        inference_model = FastLanguageModel.for_inference(base_model)
        return inference_model

    except Exception as e:
        print(f"\nError during PPO training: {e}")
        print("\nFull traceback:")
        import traceback
        print(traceback.format_exc())
        return base_model
        
def test_model(base, model, tokenizer):
    print("\nTesting the model...")
    
    # Convert both base and final models for inference
    base = FastLanguageModel.for_inference(base)
    model = FastLanguageModel.for_inference(model)
    
    base.eval()  # Ensure eval mode for base model
    model.eval()  # Ensure eval mode for final model

    test_messages = [
        {"from": "human", "value": "What is the meaning of life?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        test_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    print("Base response:")
    text_streamer = TextStreamer(tokenizer)
    _ = base.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=64,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("Model response:")
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=64,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

if __name__ == "__main__":
    # First run SFT
    sft_model, tokenizer = train_sft()

    final_model = train_ppo_custom(sft_model, tokenizer)

    # Test the final model
    test_model(sft_model, final_model, tokenizer)

    # Test the final model
    test_model(sft_model, final_model, tokenizer)
