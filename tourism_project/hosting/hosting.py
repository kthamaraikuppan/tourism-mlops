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
print(f"Listing and deleting existing content from space '{SPACE_REPO}'...")
try:
    # List all files and folders in the remote repo
    repo_files = api.list_repo_files(repo_id=SPACE_REPO, repo_type="space", token=HF_TOKEN)

    if repo_files:
        print(f"Found {len(repo_files)} existing files/folders. Deleting...")
        for file_or_folder in repo_files:
            try:
                # Attempt to delete as a file first
                api.delete_file(
                    path_in_repo=file_or_folder,
                    repo_id=SPACE_REPO,
                    repo_type="space",
                    token=HF_TOKEN,
                )
                print(f"Successfully deleted file: {file_or_folder}")
            except HfHubHTTPError as e:
                # If it's not a file, it might be a folder. Attempt to delete as a folder
                if "File not found" not in str(e):
                    try:
                        api.delete_folder(
                            path_in_repo=file_or_folder,
                            repo_id=SPACE_REPO,
                            repo_type="space",
                            token=HF_TOKEN,
                        )
                        print(f"Successfully deleted folder: {file_or_folder}")
                    except HfHubHTTPError as e_folder:
                        print(f"Error deleting {file_or_folder} (as folder): {e_folder}")
                else:
                    print(f"Error deleting {file_or_folder} (as file): {e}")
        print("Existing content deleted successfully.")
    else:
        print("No existing content found in the space.")

except HfHubHTTPError as e:
    if "not found" in str(e).lower() or "does not exist" in str(e).lower():
        print("No existing content found to delete or space is already empty (during list). ")
    else:
        print(f"Error listing content from Space: {e}")
except Exception as e:
    print(f"An unexpected error occurred during content listing/deletion: {e}")

# 3. Upload the deployment folder (app.py, Dockerfile, requirements.txt)
api.upload_folder(
    folder_path="/content/tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
print(f"Deployment files uploaded to space.")
print(f"App live at: https://huggingface.co/spaces/{SPACE_REPO}")
