This repository contains an end-to-end implementation of a custom AI reasoning engine designed for fraud analysis. The project demonstrates the full lifecycle of a Large Language Model (LLM)—from cloud-based fine-tuning to local deployment on consumer-grade hardware.

🛠️ **Key Components**
**Fine-Tuning Architecture**: Utilizes Unsloth and Llama 3.2 3B Instruct to perform memory-efficient instruction tuning on the FineTome-100k dataset.

**Precision Optimization**: Configured with a learning rate of 2e-5 and optimized LoRA adapters (Rank 16) to sharpen logical reasoning capabilities.

**Edge Deployment**: Features a GGUF quantized export (4-bit) that allows the model to run locally on CPUs using Ollama.

**LangChain Integration**: Implements a clean Python/LangChain pipeline using LCEL (LangChain Expression Language) for structured, real-time fraud risk assessment.

**Persona Engineering**: Includes a custom Modelfile that defines a specialized "Analyst" system prompt to guide the model's step-by-step reasoning logic.

📂 **Repository Structure**
**llama3_2_(1b_and_3b)_conversational.ipynb**: The cloud-based training notebook script used to fine-tune the model and export it to GGUF format.

**langchain_intro.py**: The local application script that connects the Ollama-hosted model to a LangChain inference pipeline.

**Modelfile**: The configuration blueprint used to register the fine-tuned weights and set system parameters within Ollama.

README.md: Comprehensive documentation covering the training hyperparameters, local setup instructions, and project reflections.
