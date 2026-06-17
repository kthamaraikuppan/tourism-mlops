import os
from huggingface_hub import HfApi, create_repo

# 1. Initialize the HF Token — establishes the connection with Hugging Face
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
HF_USERNAME = "kthamaraikannan"  # Set your actual Hugging Face username here
SPACE_REPO  = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi(token=HF_TOKEN)

# Create the HF Space if it doesn't exist
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except Exception:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker", # Changed from "streamlit" to "docker"
        private=False,
    )
    print(f"Space '{SPACE_REPO}' created.")

# 2. Upload the deployment folder (app.py, Dockerfile, requirements.txt)
api.upload_folder(
    folder_path="/content/tourism_project/deployment",   # local folder containing all deployment files
    repo_id=SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
print(f"Deployment files uploaded to space.")
print(f"App live at: https://huggingface.co/spaces/{SPACE_REPO}")
