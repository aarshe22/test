import os
from huggingface_hub import snapshot_download

def get_hf_token():
    token_path = "./hf-token"
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    else:
        raise FileNotFoundError("Hugging Face token file not found: ./hf-token")

hf_token = get_hf_token()

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
local_dir = "./models/mistral-7b"

if not os.path.exists(local_dir) or not os.listdir(local_dir):
    print(f"Downloading {model_id} to {local_dir} ...")
    snapshot_download(repo_id=model_id, local_dir=local_dir, token=hf_token, resume_download=True)
    print("Download complete.")
else:
    print(f"Model found in {local_dir}, skipping download.")

