import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def test_model_generation():
    print("Starting model generation test...")

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )

    # Prepare the model for inference
    model = FastLanguageModel.for_inference(model)

    # Prompts to generate
    prompts = [
        "What is the meaning of life?",
        "Explain quantum physics in simple terms."
    ]

    # Generate responses one at a time
    print("\n--- Generating Responses ---")
    for prompt in prompts:
        # Prepare inputs
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True
        ).to("cuda")

        # Generate response with extended parameters
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=128,  # Increased from 64
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,  # Reduced repetition
        )

        # Decode response, removing the original prompt and special tokens
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the beginning of the response
        response = full_response[len(prompt):].strip()

        print(f"\nPrompt: {prompt}")
        print("Generated Response:")
        print(response)

# Run the test
test_model_generation()
