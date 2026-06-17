import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# 1. Initialize the HF Token — establishes the connection with Hugging Face
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
HF_USERNAME = "kthamaraikannan"  # Set your actual Hugging Face username here
SPACE_REPO  = f"{HF_USERNAME}/tourism-package-app"

api = HfApi(token=HF_TOKEN)

# Create the HF Space if it doesn't exist
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except Exception:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        private=False,
    )
    print(f"Space '{SPACE_REPO}' created.")

# 2. Force delete existing files in the Space to ensure a clean upload
print(f"Attempting to delete specific deployment files from space '{SPACE_REPO}'...")
deployment_files_to_delete = ["app.py", "Dockerfile", "requirements.txt", ".gitattributes", "README.md"]
for file_to_delete in deployment_files_to_delete:
    try:
        api.delete_file(
            path_in_repo=file_to_delete,
            repo_id=SPACE_REPO,
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"Successfully deleted: {file_to_delete}")
    except HfHubHTTPError as e:
        if "File not found" in str(e):
            print(f"'{file_to_delete}' not found in space (already deleted or never existed).")
        else:
            print(f"Error deleting '{file_to_delete}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while deleting '{file_to_delete}': {e}")
print("Specific deployment files deletion attempt complete.")


# 3. Upload the deployment folder (app.py, Dockerfile, requirements.txt)
api.upload_folder(
    folder_path="/content/tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
print(f"Deployment files uploaded to space.")
print(f"App live at: https://huggingface.co/spaces/{SPACE_REPO}")
