from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
HF_USERNAME = "kthamaraikannan"  # Set your actual Hugging Face username here

api = HfApi(token=HF_TOKEN)

repo_id   = f"{HF_USERNAME}/tourism-data"
repo_type = "dataset"

# Step 1: Check if the dataset repo exists; create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Step 2: Upload the data folder to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Data uploaded to https://huggingface.co/datasets/{repo_id}")
