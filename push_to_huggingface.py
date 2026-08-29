# ==============================================================================
# WORKAROUND NOTE: 
# Direct cloud environment downloads can be excessively slow. Pushing the model 
# folder to the Hugging Face Hub bypasses this limitation, allowing you to pull 
# the weights down quickly from the Hub cache instead.
# 
# PREREQUISITES BEFORE RUNNING THIS SCRIPT:
# 1. Authenticate your machine via terminal using: huggingface-cli login
# 2. Create the target repository on the Hugging Face website beforehand 
#    (e.g., your-huggingface-username/fraud_model_v5_20260828)
# ==============================================================================

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Force-load .env securely from the script's exact directory for background execution
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
repo_id = "MaxwellMensah/fraud_model_v5_20260828"

api = HfApi()

print("Uploading GGUF model folder to Hugging Face...")
api.upload_folder(
    folder_path="saved_llama",
    repo_id=repo_id,
    token=HF_TOKEN,
)

print(f"Successfully uploaded! Model is ready at https://huggingface.co/{repo_id}")