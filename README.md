```markdown
# 🛡️ Custom Fraud Analysis Engine

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Fine--Tuned_With-Unsloth-FF6F00" alt="Unsloth" />
  <img src="https://img.shields.io/badge/Base_Model-Llama_3.2-purple" alt="Base Model" />
  <img src="https://img.shields.io/badge/Hugging_Face-MaxwellMensah-yellow?logo=huggingface" alt="Hugging Face" />
</p>

An end-to-end pipeline for fine-tuning **Llama 3.2** on domain-specific transaction fraud datasets using **Unsloth**. This repository covers dataset formatting, low-rank adaptation (LoRA) training, 4-bit GGUF model quantization, edge-case safety evaluation, and automated Hugging Face Hub deployment.

An end-to-end pipeline for fine-tuning **Llama 3.2** on domain-specific transaction fraud datasets using **Unsloth**. This repository covers dataset formatting, low-rank adaptation (LoRA) training, 4-bit GGUF model quantization, edge-case safety evaluation, and automated Hugging Face Hub deployment.

---

## 🔄 End-to-End Pipeline Architecture

```text
[ Dataset (.jsonl) ] ➔ [ sft_training.py ] ➔ [ outputs/ (LoRA Adapters) ]
                                                     │
[ HF Hub ]  [ push_to_huggingface.py ]  [ export_model.py ] ➔ [ saved_llama/ (Merged) ]
                                                                     │
[ Ollama Local ]  [ Modelfile ]  [ test_edge_case.py ]  [ transform_gguf.py ] (Q4_K_M)

```

---

## 🎯 Sample Model Output

The fine-tuned model enforces a strict, four-step chain-of-thought reasoning framework to evaluate transaction signals prior to issuing a final risk decision:

```text
### Input Query:
U-1193821 attempting rapid successive 4.99 USD payments (x6). IP is Proxy/Relay. Singapore, 03:45 AM. Mobile (new) used.

### Fine-Tuned Model Inference:
Step 1: Context - U-1193821 attempted a large 4.99 USD transfer from Singapore at 03:45 AM.
Step 2: Anomaly Detection - Flagged IP (Proxy/Relay), device mismatch (Mobile (new)), and unusual timing detected.
Step 3: Risk Assessment - Velocity of transactions and hidden IP suggest account takeover.
Step 4: Decision - High risk due to combined anonymized IP, new device, and shipping deviation. Transaction blocked and flagged for manual review.

```

---

## 📂 Project Structure

* **`sft_training.py`**: Handles dataset loading, ShareGPT template formatting, LoRA adapter target injection, and 4-bit SFT execution using Unsloth.
* **`export_model.py`**: Merges fine-tuned LoRA adapter weights back into the 16-bit base model and exports full local checkpoints.
* **`transform_gguf.py`**: Quantizes merged model weights into `Q4_K_M` GGUF format for CPU/edge deployment.
* **`test_edge_case.py`**: Evaluates model performance against high-value domestic transfers, foreign residential IPs, and proxy micro-transactions.
* **`push_to_huggingface.py`**: Uploads local model artifacts and GGUF files directly to Hugging Face Hub (`MaxwellMensah/fraud_model_v5_20260828`).
* **`dataset_creation.py`**: Script for generating and augmenting synthetic fraud transaction patterns.
* **`fraud_detection_dataset_V4.jsonl`**: The target fine-tuning dataset formatted with structured system, user, and assistant turns.
* **`Modelfile`**: Configuration for building local Ollama instances from exported GGUF weights.

---

## 📋 Data Format Specification

The fine-tuning dataset (`fraud_detection_dataset_V4.jsonl`) follows a structured multi-turn JSON array schema:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a fraud detection expert. Analyze transactions using step-by-step reasoning."
    },
    {
      "role": "user",
      "content": "U-8916730 attempting 12,500.00 USD transfer. IP is Corporate Network. Toronto, afternoon. Desktop (saved) used."
    },
    {
      "role": "assistant",
      "content": "Step 1 - Context: U-8916730 initiated a 12,500.00 USD payment from Toronto at afternoon.\nStep 2 - Anomaly Detection: All signals clean. IP (Corporate Network) and device (Desktop (saved)) match profile. Normal timing.\nStep 3 - Risk Assessment: Low risk due to stable location history and successful recent 2FA.\nStep 4 - Decision: Low risk - Transaction approved."
    }
  ]
}

```

---

## ⚙️ Prerequisites & Setup

1. **Install Dependencies:**
```bash
pip install unsloth python-dotenv huggingface_hub torch datasets trl transformers

```


2. **Environment Configuration:**
Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_write_token

```


3. **Hugging Face CLI Login:**
```bash
huggingface-cli login

```



---

## 🚀 Execution Workflow

Execute the scripts sequentially to fine-tune, export, test, and publish the model:

### Step 1: Fine-Tune Base Model

Runs Supervised Fine-Tuning (SFT) over 350 steps using 4-bit acceleration:

```bash
python3 sft_training.py

```

* **Input:** `fraud_detection_dataset_V4.jsonl`
* **Output:** Saved LoRA checkpoint in `outputs/checkpoint-350/`

### Step 2: Merge & Export Model Weights

Consolidates trained LoRA adapters back into full precision model weights:

```bash
python3 export_model.py

```

* **Output:** Merged 16-bit model directory saved to `saved_llama/`

### Step 3: Quantize Weights to 4-bit GGUF

Converts saved full precision weights into lightweight `Q4_K_M` GGUF binaries:

```bash
python3 transform_gguf.py

```

* **Output:** Quantized `.gguf` file generated inside `saved_llama/`

### Step 4: Run Edge-Case Performance Checks

Runs validation queries against clean enterprise transfers and hidden proxy threats:

```bash
python3 test_edge_case.py

```

### Step 5: Push Artifacts to Hugging Face Hub

Uploads model weights and GGUF binaries directly to Hugging Face for remote distribution:

```bash
python3 push_to_huggingface.py

```

* **Repository:** [MaxwellMensah/fraud_model_v5_20260828](https://www.google.com/search?q=https://huggingface.co/MaxwellMensah/fraud_model_v5_20260828)

---

## 💻 Local Ollama Deployment

Deploy your exported GGUF model locally using Ollama:

1. **Configure `Modelfile`:**
```dockerfile
FROM ./saved_llama/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.0
SYSTEM "You are a fraud detection expert. Analyze transactions using step-by-step reasoning."

```


2. **Build and Serve:**
```bash
ollama create fraud-reasoner -f Modelfile
ollama run fraud-reasoner "U-1193821 rapid successive 4.99 USD payments (x6). IP: Proxy. Singapore, 03:45 AM."

```