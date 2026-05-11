# 🛡️ Custom Fraud Analysis Engine

An end-to-end implementation of a custom-tuned **Llama 3.2 3B** reasoning engine. This project demonstrates the lifecycle of a specialized LLM from cloud-based fine-tuning to local CPU-based deployment.

---

## 🧠 Part 1: Fine-Tuning (The "Cloud" Phase)

### Model & Dataset

* **Base Model:** `unsloth/Llama-3.2-3B-Instruct`
* **Dataset:** `mlabonne/FineTome-100k` (A high-quality reasoning dataset)

### Key Configuration Changes

To increase the precision and "intelligence" of the model, the following training hyper-parameters were tuned:

* **Learning Rate:** Set to **`2e-5`** (using `adamw_8bit`) to ensure stable weight updates during the **60-step run**.
* **LoRA Configuration:** Used a **Rank (`r`) of 16** and **Alpha of 16** to target key modules (`q_proj`, `v_proj`, etc.), optimizing VRAM usage.
* **Optimization:** Employed **`train_on_responses_only`** to mask user inputs and force the model to focus purely on perfecting reasoning outputs.

### Exporting the "Brain"

After training, the model was converted to **GGUF format** to bridge the gap to local hardware:

```python
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

```

---

## 📦 Part 2: Local Deployment (The "Home" Phase)

### 1. Transferring the Model

Because the `.gguf` file is approximately **1.9GB**, it was compressed into a ZIP archive to ensure a stable download from the Kaggle environment:

```bash
zip -r fine_tuned_model.zip model_gguf

```

### 2. Creating the Ollama Specialist

To give the raw weights a specific "Job Description," an **Ollama Modelfile** was created in the local project directory:

```dockerfile
# Modelfile
FROM ./llama-3.2-3b-instruct.Q4_K_M.gguf

# Personality Injection
SYSTEM "You are a professional analyst fine-tuned for high precision. Use step-by-step logic to solve problems."

# Consistency Tuning
PARAMETER temperature 0

```

**Register the model locally:**

```bash
ollama create fraud-reasoner -f Modelfile

```

---

## 🐍 Part 3: LangChain Integration

The final application utilizes **LangChain Expression Language (LCEL)** to create a clean, streamable interface.

### `langchain_intro.py` Highlights:

* **Provider:** `langchain_ollama`
* **Output Parsing:** Uses `StrOutputParser` to return a clean string instead of a complex message object.
* **The Chain:** `prompt | llm | StrOutputParser()`

### Sample Output Analysis

**Query:** *"Analyze the fraud risk of a transaction from an unknown IP at 3 AM."*

**Result:** Instead of a simple "yes/no," the model provides a **4-step logical breakdown**:

1. **Context** Analysis
2. **Anomaly** Detection
3. **Pattern** Recognition
4. **Risk Evaluation**

---

## 🚀 How to Run

1. **Local Setup:** Install requirements via `pip install langchain-ollama langchain_core`.
2. **Ollama:** Ensure [Ollama](https://ollama.com/) is installed and running.
3. **Model Build:** Download your exported `.gguf` and run the `ollama create` command.
4. **Execute:** Run `python langchain_intro.py`.

---

## 📝 Reflections: The Cost of Abstraction

This project highlights how **LangChain** and **Ollama Modelfiles** abstract away the complexity of raw LLM interaction:

* **Visibility Cost:** We no longer see raw `<|start_header_id|>` tags or the underlying tensor math.
* **Gain:** We gain a **modular, readable pipeline** where we can swap models or prompt templates by changing a single line of code.

---

## 📂 Project Structure

* `llama3_2_conversational.py`: Kaggle/Colab Training Script
* `langchain_intro.py`: Local Inference Script
* `Modelfile`: Ollama Configuration
* `README.md`: Project Documentation
