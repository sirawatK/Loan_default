from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

import warnings
warnings.filterwarnings("ignore")

# MLflow Import
import mlflow
import mlflow.sklearn
import mlflow.xgboost

sys.path.append("/opt/airflow")

# --- CONFIGURATION ---
RAW_PATH = "/opt/airflow/data/raw/loan_default.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/clean_loan_default.csv"
TRAIN_PATH = "/opt/airflow/data/processed/train.csv"
TEST_PATH = "/opt/airflow/data/processed/test.csv"
CURRENT_DATA_PATH = "/opt/airflow/data/processed/current.csv"
MODELS_PATH = "/opt/airflow/models"

# MLflow Configuration
# Ensure this directory exists or point to your remote MLflow server
#MLFLOW_TRACKING_URI = "http://mlflow:5000" 
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost/mlflow")
EXPERIMENT_NAME = "Loan_Default_Prediction"
MODEL_REGISTRY_NAME = "LoanDefaultPredictionModel"

TARGET_COLUMN = "Default" 

try:
    from src.data.data_pipeline import build_dataset
except ImportError:
    print("Warning: src.data.data_pipeline not found. Ensure Docker volume is mounted correctly.")

default_args = {
    'owner': 'G.Loan_Default',
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}

# --- HELPER FUNCTIONS ---

def check_raw_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print("Raw data loaded with shape:", df.shape)

def run_cleaning():
    print("Starting data cleaning...")
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    build_dataset(RAW_PATH, PROCESSED_PATH)
    print("Saved cleaned file:", PROCESSED_PATH)

def split_data():
    df = pd.read_csv(PROCESSED_PATH)
    from sklearn.model_selection import train_test_split
    
    # Ensure target column exists before split
    if TARGET_COLUMN not in df.columns:
         raise ValueError(f"Target '{TARGET_COLUMN}' not found. Available: {df.columns}")
    train, temp_data = train_test_split(df, test_size=0.2, random_state=42)
    test, current = train_test_split(df, test_size=0.5, random_state=42)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    current.to_csv(CURRENT_DATA_PATH, index=False)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Current shape:", current.shape)

def _load_train_data():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found at {TRAIN_PATH}")
    
    train_df = pd.read_csv(TRAIN_PATH)
    
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    return X_train, y_train

def _load_test_data():
    """Helper to load Test data for evaluation"""
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Test file not found at {TEST_PATH}")
    
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_test, y_test

# --- TRAINING FUNCTIONS ---

def train_evaluate():
    import joblib
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
    
    X_train, y_train = _load_train_data()
    X_test, y_test = _load_test_data()
    models = [
    (
        "XGBoost", 
        {
            "objective":'binary:logistic',
            "eval_metric":'logloss',
            "use_label_encoder":False,
            "random_state":42,
            "enable_categorical":True
        },
        XGBClassifier(), 
        (X_train, y_train),
        (X_test, y_test),
        "xgboost_model.pkl"
        
    ),
    (
        "Logistic Regression", 
        {"random_state": 42, "max_iter": 2000},
        LogisticRegression(), 
        (X_train, y_train),
        (X_test, y_test),
        "log_reg_model.pkl"
    ),
    (
        "Random Forest", 
        {"random_state": 42, "n_estimators": 1000, "max_depth": 6},
        RandomForestClassifier(), 
        (X_train, y_train),
        (X_test, y_test),
        "random_forest_model_depth6.pkl"
    )
    ]

    best_rec = 0.0
    reports = []
    current_run_id = None
    current_run_name = None
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("MLFLOW_TRACKING_URI:",MLFLOW_TRACKING_URI)
    
    for model_name, params, model, train_set, test_set,model_file_name in models:
        print(f"Training {model_name}...")
        X_train = train_set[0]
        y_train = train_set[1]
        X_test = test_set[0]
        y_test = test_set[1]

        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports.append(report)
        save_path = os.path.join(MODELS_PATH, model_file_name)
        joblib.dump(model, save_path)
        print(f"{model_name} model saved to {save_path}")


# --- EVALUATION & LOGGING FUNCTION ---
    for i, element in enumerate(models):
        model_name = element[0]
        params = element[1]
        model = element[2]
        model_file_name = element[5]
        report = reports[i]
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%H%M')}") as run:
            current_run_id = run.info.run_id
            current_run_name = run.data.tags.get('mlflow.runName')
            mlflow.log_params(params)
            mlflow.log_metrics({
                'accuracy': report['accuracy'],
                'recall_class_1': report['1']['recall'],
                'recall_class_0': report['0']['recall'],
                'f1_score_macro': report['macro avg']['f1-score']
            })

            # Log the actual model file to MLflow artifacts
            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
                
            print(f"Logged {model_name}: rec_score={report['1']['recall']:.4f}")

        # Check if this is the best model
        if report['1']['recall'] > best_rec:
            best_rec = report['1']['recall']
            best_model_name = model_name
            best_run_id = current_run_id
            best_run_name = current_run_name
            best_model_file_name = model_file_name

    # 4. Output Winner
    print("-" * 30)
    print(f"🏆 BEST MODEL: {best_model_name} with Recall: {best_rec:.4f}")
    print("-" * 30)
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # 1. Register the best model
        model_info = mlflow.register_model(
            model_uri=model_uri, 
            name=MODEL_REGISTRY_NAME
        )
        print(f"Model successfully registered as new version '{model_info.version}' of '{MODEL_REGISTRY_NAME}'")
        
        # 2. Tag the newly registered version as 'Champion'
        # Note: We use 'set_model_version_tag' for custom tags like "Champion" 
        # or 'transition_model_version_stage' to move it to 'Production' (standard stage)
        client.set_model_version_tag(
            name=MODEL_REGISTRY_NAME,
            version=model_info.version,
            key="Status",
            value="Production"
        )
        print(f"Model version {model_info.version} tagged as 'Production'.")
        
    except Exception as e:
        print(f"Error registering or tagging model: {e}")

    best_model_info = {
        "model": best_model_name,
        "file": best_model_file_name,
        "recall": best_rec
    }

    # Write the dictionary to the file as JSON
    with open(os.path.join(MODELS_PATH, "best_model_info.json"), "w") as f:
        json.dump(best_model_info, f, indent=4)

# --- DAG DEFINITION ---

with DAG(
    dag_id="loan_default_pipeline_v4_mlflow",
    description="Loan Default Pipeline with MLflow Evaluation",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")
    
    verify_raw = PythonOperator(
        task_id="check_raw_data", 
        python_callable=check_raw_data
    )
    
    clean_and_engineer = PythonOperator(
        task_id="clean_data", 
        python_callable=run_cleaning
    )
    
    split = PythonOperator(
        task_id="train_test_split", 
        python_callable=split_data
    )

    # Evaluation Task
    evaluate = PythonOperator(
        task_id="train_evaluate_and_log_models",
        python_callable=train_evaluate
    )

    end = EmptyOperator(task_id="end")

    # --- DEPENDENCIES ---
    
    # Data Prep Flow
    start >> verify_raw >> clean_and_engineer >> split
    
    split >> evaluate

    # End
    evaluate >> end