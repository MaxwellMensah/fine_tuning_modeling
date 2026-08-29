from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/checkpoint-350",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Converts and creates the 'my_llama_gguf' folder
model.save_pretrained_gguf("saved_llama", tokenizer, quantization_method = "q4_k_m")