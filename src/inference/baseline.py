import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID="Qwen/Qwen3-0.6B"

def main():
    print("="*50)
    print("QWEN3-0.6B BASELINE INFERENCE")
    print("="*50)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading tokenizer...")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loding model..")

    model= AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        use_safetensors=True,
    )


    model.eval()

    print("\nModel loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    messages=[
        {
            "role":"user",
            "content":"Explain what supervised fine-tuning is in simple terms."
        }
    ]

    inputs=tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)



    print("\nGenerating response....")
    # print("EOS token ID:", model.generation_config.eos_token_id)
    # print("PAD token ID:", model.generation_config.pad_token_id)
    # print("Input token count:", inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        outputs=model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
            # return_dict_in_generate=True,
        )

    # generated_tokens=outputs.sequences[0][inputs["input_ids"].shape[-1]:]
    generated_tokens=outputs[0][inputs["input_ids"].shape[-1]:]

    # print("\nGenerated token IDs:",generated_tokens.tolist())
    # print("Generated token count:", len(generated_tokens))

    # print("\nIndividual tokens:")
    # for token_id in generated_tokens.tolist():
    #     print(token_id,"->", repr(tokenizer.decode([token_id])))

    response=tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    # print("\nFinal response:", response)

    print("\n"+"="*50)
    print("MODEL RESPONSE")
    print("="*50)
    print(response)

    print("\n"+"="*50)
    print("GPU MEMORY")
    print("="*50)

    allocated=torch.cuda.memory_allocated(0)/1024**3
    reserved=torch.cuda.memory_reserved(0)/1024**3

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")


if __name__=="__main__":
    main()
