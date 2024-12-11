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

    # Reduce batch size for debugging
    batch_size = 1

    # Add debugging prints for tensors
    def debug_tensor(name, tensor):
        if tensor is not None:
            print(f"{name} shape: {tensor.shape}")
        else:
            print(f"{name} is None")

    try:
        for epoch in range(1):
            print(f"\nEpoch {epoch + 1}/1")
            # Simulate dataset loading and model output
            # Add debugging prints for inputs and outputs
            query = torch.rand(batch_size, 32)
            cos = torch.rand(batch_size, 32)
            sin = torch.rand(batch_size, 32)

            debug_tensor("Query", query)
            debug_tensor("Cos", cos)
            debug_tensor("Sin", sin)

            # Placeholder for actual PPO logic
            # Simulate tensor operations causing errors
            try:
                Q = query * cos
                debug_tensor("Q", Q)

            except RuntimeError as e:
                print("Error during tensor operation:", e)
                raise

        print("PPO Training completed!")
    except Exception as e:
        print("Error during PPO training:", e)
        raise

def test_model(base, model, tokenizer):
    print("\nTesting the model...")

    if base is None:
        print("Base model is None. Skipping test.")
        return

    if model is None:
        print("Final model is None. Skipping test.")
        return

    # Convert both base and final models for inference
    try:
        base = FastLanguageModel.for_inference(base)
        model = FastLanguageModel.for_inference(model)
    except AttributeError as e:
        print("Error converting model for inference:", e)
        return

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

    if sft_model is not None and tokenizer is not None:
        final_model = train_ppo_custom(sft_model, tokenizer)

        # Test the final model
        test_model(sft_model, final_model, tokenizer)
