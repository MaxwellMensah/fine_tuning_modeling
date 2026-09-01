import torch
from datasets import load_dataset

# always import unsloth first
from unsloth import FastLanguageModel
# then import TRL
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only


def log_step(step_name):
    """Prints a clear separator to show exactly which phase is starting."""
    print(f"\n{'=' * 60}")
    print(f"🚀 [STAGE]: {step_name}")
    print(f"{'=' * 60}")


def print_memory_usage(context="Current"):
    """Logs the exact GPU memory allocated and reserved at a specific point in time."""
    if not torch.cuda.is_available():
        print(f"➔ [VRAM Status | {context}] CUDA not available. Running on CPU.")
        return

    reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    allocated = round(torch.cuda.memory_allocated() / 1024**3, 3)
    print(
        f"➔ [VRAM Status | {context}] Reserved: {reserved} GB | Allocated: {allocated} GB"
    )


if __name__ == "__main__":
    # SYSTEM INITIALIZATION
    log_step("SYSTEM INITIALIZATION")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"Detected GPU: {gpu_stats.name} | Max Memory: {max_memory} GB")
    else:
        max_memory = 0
        print("No GPU detected.")

    print_memory_usage("Baseline before loading model")

    # LOAD BASE MODEL & TOKENIZER
    log_step("LOADING BASE MODEL & TOKENIZER")
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print("Base model and tokenizer loaded successfully.")
    print_memory_usage("After loading 4-bit base model")

    # INJECT LORA ADAPTERS
    log_step("INJECTING LORA ADAPTERS")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("LoRA adapters successfully attached to target modules.")
    print_memory_usage("After attaching LoRA adapters")

    # DATASET PREPARATION (Separate Train & Validation files from decoupled directory)
    log_step("DATASET PREPARATION (TRAIN & VALIDATION)")

    dataset = load_dataset(
        "json",
        data_files={
            "train": "dataset_creation/train_2400.jsonl",
            "validation": "dataset_creation/val_600.jsonl",
        },
    )
    print(f"Loaded train dataset: {len(dataset['train'])} examples.")
    print(f"Loaded validation dataset: {len(dataset['validation'])} examples.")

    def format_prompts(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    print("Applying chat template formatting map to train and validation splits...")
    dataset = dataset.map(format_prompts, batched=True)

    print("Datasets formatted successfully using native chat template.")
    print_memory_usage("After dataset processing")

    # TRAINING SETUP & EXECUTION
    log_step("TRAINING SETUP & EXECUTION")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir="outputs_v7",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            max_steps=350,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            dataset_text_field="text",
            max_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,
        ),
    )

    # MASK INSTRUCTION TOKENS: Train ONLY on assistant response reasoning
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    start_gpu_memory = (
        round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        if torch.cuda.is_available()
        else 0
    )
    print_memory_usage("Immediately before trainer.train()")

    trainer_stats = trainer.train()

    # POST-TRAINING MEMORY ANALYTICS
    log_step("POST-TRAINING MEMORY ANALYTICS")

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = (
            round(used_memory / max_memory * 100, 3) if max_memory > 0 else 0
        )
        lora_percentage = (
            round(used_memory_for_lora / max_memory * 100, 3) if max_memory > 0 else 0
        )

        print(
            f"Time: {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
        )
        print(f"Peak Reserved Memory: {used_memory} GB")
        print(f"Peak Reserved Memory for Training (Delta): {used_memory_for_lora} GB")
        print(f"Total VRAM Utilization: {used_percentage} % of {max_memory} GB limit")
        print(
            f"Training VRAM Utilization: {lora_percentage} % of {max_memory} GB limit"
        )

    # SAVING FINAL ADAPTER CHECKPOINT
    log_step("SAVING FINAL CHECKPOINT")
    model.save_pretrained("outputs_v7/final_checkpoint")
    tokenizer.save_pretrained("outputs_v7/final_checkpoint")
    print("Training complete. LoRA adapter saved to outputs_v7/final_checkpoint.")