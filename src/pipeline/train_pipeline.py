import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Retrieve secrets from environment variables (set in GitHub Actions)
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it as a GitHub Secret.")
if not HF_USERNAME:
    raise ValueError("HF_USERNAME not found in environment variables. Please set it as a GitHub Secret.")

DATA_REPO   = f"{HF_USERNAME}/tourism-data"
MODEL_REPO  = f"{HF_USERNAME}/tourism-package-model"

# 1. Set MLflow tracking
# For GitHub Actions, MLflow tracking might be configured differently or skipped if only interested in model artifact
mlflow.set_tracking_uri("http://localhost:5000") # Placeholder: adjust if using remote MLflow server
mlflow.set_experiment("tourism-Package-Prediction-Experiment")

# 2. Initialize HF API
api = HfApi(token=HF_TOKEN)

# 3. Read csv files from Hugging Face by downloading them first
print("Attempting to download data files from Hugging Face Hub...")
Xtrain_local_path = hf_hub_download(repo_id=DATA_REPO, filename="Xtrain.csv", repo_type="dataset", token=HF_TOKEN)
Xtest_local_path  = hf_hub_download(repo_id=DATA_REPO, filename="Xtest.csv", repo_type="dataset", token=HF_TOKEN)
ytrain_local_path = hf_hub_download(repo_id=DATA_REPO, filename="ytrain.csv", repo_type="dataset", token=HF_TOKEN)
ytest_local_path  = hf_hub_download(repo_id=DATA_REPO, filename="ytest.csv", repo_type="dataset", token=HF_TOKEN)

# 4. Load datasets from local paths
X_train = pd.read_csv(Xtrain_local_path)
X_test  = pd.read_csv(Xtest_local_path)
y_train = pd.read_csv(ytrain_local_path).squeeze()
y_test  = pd.read_csv(ytest_local_path).squeeze()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 5. Separate numerical and categorical variables
num_features = X_train.select_dtypes(include=np.number).columns.tolist()
cat_features = X_train.select_dtypes(include="object").columns.tolist()

# 6. Preprocess using make_column_transformer
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = make_column_transformer(
    (num_transformer, num_features),
    (cat_transformer, cat_features),
)

# 7. Base model
base_model = RandomForestClassifier(random_state=42)

# 8. Hyperparameter grid
param_grid = {
    "classifier__n_estimators":     [100],
    "classifier__max_depth":        [None],
    "classifier__min_samples_split":[2],
} # Using the best params found previously to speed up GitHub Actions

# 9. Full pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   base_model),
])

# MLflow run
with mlflow.start_run():

    # 10. GridSearchCV
    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 11. Log parameters
    best_params = grid_search.best_params_
    mlflow.log_params(best_params)
    print("Best params:", best_params)

    # 12. Best model
    best_model = grid_search.best_estimator_

    # 13. Predictions on train and test
    y_train_pred  = best_model.predict(X_train)
    y_test_pred   = best_model.predict(X_test)
    y_test_proba  = best_model.predict_proba(X_test)[:, 1]

    # 14. Evaluate performance
    train_acc  = accuracy_score(y_train, y_train_pred)
    test_acc   = accuracy_score(y_test,  y_test_pred)
    precision  = precision_score(y_test, y_test_pred)
    recall     = recall_score(y_test,    y_test_pred)
    f1         = f1_score(y_test,        y_test_pred)
    roc_auc    = roc_auc_score(y_test,   y_test_proba)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC AUC        : {roc_auc:.4f}")

    # 15. Log metrics on MLflow
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1,
        "roc_auc":        roc_auc,
    })

    # 16. Save model locally for subsequent upload
    os.makedirs("model_output", exist_ok=True) # Create a directory for model output
    model_path   = "model_output/best_model.joblib"
    metrics_path = "model_output/metrics.json"
    joblib.dump(best_model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"params": best_params, "metrics": {
            "train_accuracy": train_acc, "test_accuracy": test_acc,
            "precision": precision, "recall": recall,
            "f1_score": f1, "roc_auc": roc_auc,
        }}, f, indent=2)
    print("Model and metrics saved locally for upload.")

    # 17. Log model to MLflow
    mlflow.sklearn.log_model(best_model, "best_model", skops_trusted_types=['numpy.dtype', 'sklearn.compose._column_transformer._RemainderColsList'])
