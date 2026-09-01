import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs_v7/checkpoint-350",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

edge_cases = [
    "U-8916730 attempting 12,500.00 USD transfer. IP is Corporate Network. Toronto, afternoon. Desktop (saved) used.",
    "U-4402198 sending 45.00 GBP at morning. IP is UK Residential. Card Issued: UK. Mobile (saved) used.",
    "U-1193821 attempting rapid successive 4.99 USD payments (x6). IP is Proxy/Relay. Singapore, 03:45 AM. Mobile (new) used."
]

for i, case in enumerate(edge_cases, 1):
    messages = [
        {"role": "system", "content": "You are a fraud detection expert system analyzing transaction risk."},
        {"role": "user", "content": case}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEvaluating Edge Case {i}...")
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.1,
        min_p=0.1
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("=" * 60)