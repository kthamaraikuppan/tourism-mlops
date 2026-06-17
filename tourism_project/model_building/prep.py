import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
HF_USERNAME = "kthamaraikannan"

# 1. Prepare for data manipulation
api = HfApi(token=HF_TOKEN)

# 2. Load the dataset directly from Hugging Face
DATASET_PATH = f"hf://datasets/{HF_USERNAME}/tourism-data/tourism.csv"
df = pd.read_csv(DATASET_PATH, storage_options={"token": HF_TOKEN})
print("Loaded dataset from HF:", df.shape)

# 3. Clean
df = df.drop_duplicates()
df.drop(columns=[c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns], inplace=True)
df["Gender"]        = df["Gender"].str.strip().replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].str.strip().replace("Unmarried", "Single")

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for c in num_cols: df[c].fillna(df[c].median(), inplace=True)
for c in cat_cols: df[c].fillna(df[c].mode()[0], inplace=True)
print("Cleaned data:", df.shape)

# 4. Separate target and features
TARGET = "ProdTaken"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 5. Separate categorical and numerical columns
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(include="object").columns.tolist()
print("Numeric features:", num_features)
print("Categorical features:", cat_features)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 7. Save split data locally
os.makedirs("tourism_project/data", exist_ok=True)
X_train.to_csv("tourism_project/data/Xtrain.csv", index=False)
X_test.to_csv("tourism_project/data/Xtest.csv",   index=False)
y_train.to_csv("tourism_project/data/ytrain.csv", index=False)
y_test.to_csv("tourism_project/data/ytest.csv",   index=False)
print("Split data saved locally.")

# 8. Upload split files to Hugging Face dataset repo
repo_id = f"{HF_USERNAME}/tourism-data"
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{fname}",
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"Uploaded {fname} → {repo_id}")
print("Data preparation complete.")
