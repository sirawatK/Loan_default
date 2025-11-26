from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import mlflow

# --- CONFIGURATION ---
# Adjust these paths to match your actual file locations
TEST_PATH = "/opt/airflow/data/processed/test.csv"
CURRENT_DATA_PATH = "/opt/airflow/data/current_edit_drifted.csv"
EVIDENTLY_REPORTS_PATH = "/opt/airflow/evidently_reports"
os.makedirs(EVIDENTLY_REPORTS_PATH, exist_ok=True)

TARGET_COLUMN = "Default"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost/mlflow")
EXPERIMENT_NAME = "Loan_Default_Drift_Monitoring"
RETRAIN_DAG_ID = "loan_default_pipeline_v4_mlflow"

default_args = {
    'owner': 'G.Loan_Default',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_target_drift(ti, **kwargs):
    """
    Loads reference and current data, calculates Target Drift using Evidently,
    and pushes the drift status to XCom.
    """
    # 1. Load Data
    if not os.path.exists(TEST_PATH) or not os.path.exists(CURRENT_DATA_PATH):
        raise FileNotFoundError(f"Data files not found. Checked: {TEST_PATH} and {CURRENT_DATA_PATH}")

    print("Loading reference (test) and current data...")
    reference_data = pd.read_csv(TEST_PATH)
    current_data = pd.read_csv(CURRENT_DATA_PATH)

    # 2. Validate Target Column Exists
    if TARGET_COLUMN not in reference_data.columns or TARGET_COLUMN not in current_data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from one of the datasets.")

    # 3. Configure Evidently Report
    try:
        from evidently.report import Report
        from evidently.metric_preset import TargetDriftPreset
    except ImportError:
        # Pushing a failure status if Evidently is missing
        print("Evidently is not installed. Pushing 'drift_detected' to force retrain/alert.")
        return "drift_detected"

    print(f"Calculating Target Drift for column: {TARGET_COLUMN}")
    
    column_mapping = {
        'target': TARGET_COLUMN,
    }

    report = Report(metrics=[
        TargetDriftPreset()
    ])

    try:
        report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    except Exception as e:
        print(f"Evidently run failed: {e}. Pushing 'drift_detected' as safeguard.")
        return "drift_detected"

    # 4. Save Reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_path = os.path.join(EVIDENTLY_REPORTS_PATH, f"target_drift_{timestamp}.html")
    json_path = os.path.join(EVIDENTLY_REPORTS_PATH, f"target_drift_{timestamp}.json")

    report.save_html(html_path)
    report.save_json(json_path)
    print(f"Reports saved to:\n HTML: {html_path}\n JSON: {json_path}")

    # 5. Log to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    drift_detected = False
    
    with mlflow.start_run(run_name=f"Target_Drift_Check_{timestamp}"):
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(json_path)
        
        # 6. Parse JSON to check for drift (Simplified and Robust Check)
        report_data = json.loads(report.json())
        
        try:
            # TargetDriftPreset is simple; usually, the top-level metric contains the detection status
            # We look for the general 'drift_detected' flag which is often summarized at a high level.
            # Otherwise, we check the specific Target column.
            
            # Simplified Check 1: Check the overall summary metric (if present)
            metrics = report_data.get('metrics', [])
            
            # Check for the TargetDrift metric specifically
            target_drift_metric = next((m for m in metrics if 'TargetDrift' in m['metric_id']), None)
            
            if target_drift_metric:
                # Check for direct 'drift_detected' flag within the target drift result
                if target_drift_metric.get('result', {}).get('drift_detected') is True:
                    drift_detected = True
            
            if not drift_detected:
                # Simplified Check 2: Check the target column specifically 
                # (This path structure is robust for most Evidenty reports)
                for metric in metrics:
                    result = metric.get('result', {})
                    if 'drift_by_columns' in result:
                        target_drift_info = result['drift_by_columns'].get(TARGET_COLUMN, {})
                        if target_drift_info.get('drift_detected') is True:
                            drift_detected = True
                            break
            
            print(f"Drift Detected: {drift_detected}")
            mlflow.log_metric("drift_detected", int(drift_detected))
            
        except Exception as e:
            # If JSON parsing fails, log error but assume no drift to prevent unnecessary trigger 
            # unless the error is the main cause of the pipeline failure.
            print(f"Error parsing drift metrics: {e}. Defaulting to no drift.")
            drift_detected = False # Safety default

    # Push result to XCom
    return "drift_detected" if drift_detected else "no_drift"

def branch_decision(ti, **kwargs):
    """
    Decides which task to run based on XCom result from check_target_drift.
    """
    drift_status = ti.xcom_pull(task_ids='check_target_drift')
    print(f"Branching based on status: {drift_status}")
    
    if drift_status == "drift_detected":
        return "trigger_retrain_pipeline"
    else:
        return "end"

# --- DAG DEFINITION ---

with DAG(
    dag_id="loan_default_target_drift_check",
    description="Checks specifically for drift in the Target column and triggers retrain if detected",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")

    # 1. Calculate Drift
    drift_check_task = PythonOperator(
        task_id="check_target_drift",
        python_callable=check_target_drift,
        provide_context=True
    )

    # 2. Decide Branch
    branch_task = BranchPythonOperator(
        task_id="branch_task",
        python_callable=branch_decision,
        provide_context=True
    )

    # 3a. Trigger Retraining (if drift detected)
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain_pipeline",
        trigger_dag_id=RETRAIN_DAG_ID,
        conf={"message": "Triggered by Target Drift Check"},
        wait_for_completion=False # Fire and forget
    )

    # 3b. End (if no drift)
    end = EmptyOperator(task_id="end")

    # Flow
    start >> drift_check_task >> branch_task
    branch_task >> trigger_retrain 
    branch_task >> end