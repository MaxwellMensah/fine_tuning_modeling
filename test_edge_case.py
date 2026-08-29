import torch
from unsloth import FastLanguageModel

# Load your fine-tuned checkpoint from outputs/
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/checkpoint-350", 
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

                    ### Instruction:
                    {}

                    ### Response:
                    {}
                """

edge_cases = [
    "You are a fraud detection expert. Analyze transactions using step-by-step reasoning.\n\nU-8916730 attempting 12,500.00 USD transfer. IP is Corporate Network. Toronto, afternoon. Desktop (saved) used.",
    "You are a fraud detection expert. Analyze transactions using step-by-step reasoning.\n\nU-4402198 sending 45.00 GBP at morning. IP is UK Residential. Card Issued: UK. Mobile (saved) used.",
    "You are a fraud detection expert. Analyze transactions using step-by-step reasoning.\n\nU-1193821 attempting rapid successive 4.99 USD payments (x6). IP is Proxy/Relay. Singapore, 03:45 AM. Mobile (new) used."
]

for i, case in enumerate(edge_cases, 1):
    inputs = tokenizer(
        [prompt_template.format(case, "")], 
        return_tensors = "pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEvaluating Edge Case {i}...")
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens = True)[0])
    print("=" * 60)