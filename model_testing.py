import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

def test_model_generation():
    print("Starting model generation test...")

    # Detailed model loading
    try:
        # Load the model with explicit configurations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )

        # Print initial model and tokenizer information
        print("\n--- Model and Tokenizer Information ---")
        print(f"Model Type: {type(model)}")
        print(f"Tokenizer Type: {type(tokenizer)}")

        # Prepare the model for inference
        model = FastLanguageModel.for_inference(model)

        # Prepare input
        prompts = [
            "What is the meaning of life?",
            "Explain quantum physics in simple terms."
        ]

        # Tokenize inputs
        inputs = tokenizer(
            prompts, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")

        print("\n--- Input Tensor Details ---")
        for key, value in inputs.items():
            print(f"{key} - Shape: {value.shape}, Dtype: {value.dtype}")

        # Prepare text streamer for output
        streamer = TextStreamer(tokenizer)

        # Attempt generation with comprehensive parameters
        print("\n--- Generating Responses ---")
        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                streamer=streamer
            )

            # Decode and print responses
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("\nGenerated Responses:")
            for i, response in enumerate(responses):
                print(f"Prompt {i+1}: {response}")

        except Exception as gen_error:
            print(f"\nGeneration Error: {gen_error}")
            import traceback
            traceback.print_exc()

    except Exception as load_error:
        print(f"\nModel Loading Error: {load_error}")
        import traceback
        traceback.print_exc()

# Run the test
test_model_generation()
