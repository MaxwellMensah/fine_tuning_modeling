from unsloth import FastLanguageModel

# Load the trained checkpoint from outputs_v7/ folder
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_v7/checkpoint-350",  # points to your v7 step checkpoint
    max_seq_length=2048,
    load_in_4bit=True,
)

# Save as a standalone 16-bit merged model into the decoupled directory
model.save_pretrained_merged(
    "fraud_model_v7", tokenizer, save_method="merged_16bit"
)

print("Saved clean, merged model to 'fraud_model_v7' folder!")