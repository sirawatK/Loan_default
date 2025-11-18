from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys


sys.path.append("/opt/airflow/src")

from src.data.data_pipeline import build_dataset


# PATHS for Docker Airflow
RAW_PATH = "/opt/airflow/data/raw/loan_default.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/clean_loan_default.csv"
TRAIN_PATH = "/opt/airflow/data/processed/train.csv"
TEST_PATH = "/opt/airflow/data/processed/test.csv"

# Import data cleaning pipeline
from src.data.data_pipeline import build_dataset

default_args = {
    'owner': 'G.Loan_Default',
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}

def check_raw_data():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print("Raw data loaded with shape:", df.shape)

def run_cleaning():
    print("Starting data cleaning...")
    build_dataset(RAW_PATH, PROCESSED_PATH)
    print("Saved cleaned file:", PROCESSED_PATH)

def split_data():
    df = pd.read_csv(PROCESSED_PATH)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

with DAG(
    dag_id="loan_default_data_pipeline",
    description="CPE393 Data Pipeline for Loan Default Prediction",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
):
    start = EmptyOperator(task_id="start")
    verify_raw = PythonOperator(task_id="check_raw_data", python_callable=check_raw_data)
    clean_and_engineer = PythonOperator(task_id="clean_data", python_callable=run_cleaning)
    split = PythonOperator(task_id="train_test_split", python_callable=split_data)
    end = EmptyOperator(task_id="end")

    start >> verify_raw >> clean_and_engineer >> split >> end
