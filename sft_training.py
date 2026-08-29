import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def log_step(step_name):
    """Prints a clear separator to show exactly which phase is starting."""
    print(f"\n{'='*60}")
    print(f"🚀 [STAGE]: {step_name}")
    print(f"{'='*60}")

def print_memory_usage(context="Current"):
    """Logs the exact GPU memory allocated and reserved at a specific point in time."""
    if not torch.cuda.is_available():
        print(f"➔ [VRAM Status | {context}] CUDA not available. Running on CPU.")
        return
    
    reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    allocated = round(torch.cuda.memory_allocated() / 1024**3, 3)
    print(f"➔ [VRAM Status | {context}] Reserved: {reserved} GB | Allocated: {allocated} GB")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. SYSTEM INITIALIZATION
    # ---------------------------------------------------------
    log_step("SYSTEM INITIALIZATION")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"Detected GPU: {gpu_stats.name} | Max Memory: {max_memory} GB")
    else:
        max_memory = 0
        print("No GPU detected.")
    
    print_memory_usage("Baseline before loading model")

    # ---------------------------------------------------------
    # 2. LOAD BASE MODEL & TOKENIZER
    # ---------------------------------------------------------
    log_step("LOADING BASE MODEL & TOKENIZER")
    max_seq_length = 2048
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = None, 
        load_in_4bit = True,
    )
    
    print("Base model and tokenizer loaded successfully.")
    print_memory_usage("After loading 4-bit base model")

    # ---------------------------------------------------------
    # 3. INJECT LORA ADAPTERS
    # ---------------------------------------------------------
    log_step("INJECTING LORA ADAPTERS")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    
    print("LoRA adapters successfully attached to target modules.")
    print_memory_usage("After attaching LoRA adapters")

    # ---------------------------------------------------------
    # 4. DATASET PREPARATION
    # ---------------------------------------------------------
    log_step("DATASET PREPARATION")
    
    prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    # Load local conversational JSONL dataset
    dataset = load_dataset("json", data_files="fraud_detection_dataset_V4.jsonl", split="train")
    print(f"Loaded {len(dataset)} examples. Detected columns: {dataset.column_names}")

    EOS_TOKEN = tokenizer.eos_token
    def format_prompts(examples):
        conversations_batch = examples["conversations"]
        texts = []
        
        for conv in conversations_batch:
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            
            # Extract roles and content from the conversational turns
            for turn in conv:
                role = turn.get("role")
                content = turn.get("content")
                
                if role == "system":
                    system_msg = content
                elif role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content
            
            # Combine system prompt context and user prompt into a single instruction block
            if system_msg:
                instruction = f"{system_msg}\n\n{user_msg}"
            else:
                instruction = user_msg
            
            text = prompt_template.format(instruction, assistant_msg) + EOS_TOKEN
            texts.append(text)
            
        return { "text" : texts }

    print("Applying formatting map to conversational turns...")
    dataset = dataset.map(format_prompts, batched = True)
    
    print("Dataset formatted and EOS tokens appended.")
    print_memory_usage("After dataset processing")

    # ---------------------------------------------------------
    # 5. TRAINING SETUP & EXECUTION
    # ---------------------------------------------------------
    log_step("TRAINING SETUP & EXECUTION")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 20,
            max_steps = 350,  # Covers roughly 1.1 epochs of your 2,500 rows
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 25,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3) if torch.cuda.is_available() else 0
    print_memory_usage("Immediately before trainer.train()")

    trainer_stats = trainer.train()

    # ---------------------------------------------------------
    # 6. POST-TRAINING MEMORY ANALYTICS
    # ---------------------------------------------------------
    log_step("POST-TRAINING MEMORY ANALYTICS")
    
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3) if max_memory > 0 else 0

        print(f"Time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak Reserved Memory: {used_memory} GB")
        print(f"Peak Reserved Memory for Training (Delta): {used_memory_for_lora} GB")
        print(f"Total VRAM Utilization: {used_percentage} % of {max_memory} GB limit")
        print(f"Training VRAM Utilization: {lora_percentage} % of {max_memory} GB limit")

    # ---------------------------------------------------------
    # 7. INFERENCE TESTING
    # ---------------------------------------------------------
    log_step("INFERENCE TESTING")
    
    FastLanguageModel.for_inference(model)
    print_memory_usage("After switching to inference mode (Gradient memory released)")

    sample_instruction = "You are a fraud detection expert. Analyze transactions using step-by-step reasoning.\n\nU-2604379 attempting 2360 NGN transfer. IP is VPN Datacenter. Sydney, late night. Mobile (new) used."
    inputs = tokenizer(
    [
        prompt_template.format(
            sample_instruction, 
            "", 
        )
    ], return_tensors = "pt").to("cuda" if torch.cuda.is_available() else "cpu")

    print("Generating response...\n")
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    print(tokenizer.batch_decode(outputs, skip_special_tokens = True)[0])

    print_memory_usage("After inference generation")