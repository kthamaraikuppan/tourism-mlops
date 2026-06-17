import os
from huggingface_hub import HfApi, create_repo

# Retrieve secrets from environment variables (set in GitHub Actions)
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it as a GitHub Secret.")
if not HF_USERNAME:
    raise ValueError("HF_USERNAME not found in environment variables. Please set it as a GitHub Secret.")

MODEL_REPO  = f"{HF_USERNAME}/tourism-package-model"

api = HfApi(token=HF_TOKEN)

# Create the model repo on Hugging Face if it doesn't exist
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists.")
except Exception:
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repo '{MODEL_REPO}' created.")

# Define the local path where the model and metrics are saved by train_pipeline.py
model_output_dir = "model_output"
model_path       = os.path.join(model_output_dir, "best_model.joblib")
metrics_path     = os.path.join(model_output_dir, "metrics.json")

# Upload model and metrics to Hugging Face model hub
for fpath, fname in [(model_path, "best_model.joblib"), (metrics_path, "metrics.json")]:
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"Uploaded {fname} \u2192 {MODEL_REPO}")
    else:
        print(f"Warning: {fpath} not found. Skipping upload.")

print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
