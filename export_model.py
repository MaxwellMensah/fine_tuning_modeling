from unsloth import FastLanguageModel

# 1. Load the trained checkpoint from outputs/
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/checkpoint-350", # Points to your saved step
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Save as a standalone 16-bit merged model
model.save_pretrained_merged("final_llama_model", tokenizer, save_method = "merged_16bit")

print("Saved clean, merged model to 'final_llama_model' folder!")