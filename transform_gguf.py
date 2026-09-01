from unsloth import FastLanguageModel

max_seq_length = 2048

print("Loading fine-tuned checkpoint for GGUF conversion...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_v7/checkpoint-350",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

print(
    "Converting and quantizing model to Q8_0 GGUF inside"
    " fraud_model_v7_gguf/..."
)
model.save_pretrained_gguf("fraud_model_v7_gguf", tokenizer, quantization_method="q8_0")
print("GGUF transformation complete. Model ready for Ollama.")