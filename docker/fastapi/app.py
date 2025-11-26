from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import io
import os
import json
import gzip
import shutil
import sys
import warnings
import traceback

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add /app to sys.path so we can import src
sys.path.append("/app")

try:
    from src.data.data_pipeline import build_dataset
except ImportError:
    print("Warning: Could not import src.data.data_pipeline. Ensure 'src' is mounted to /app/src")

app = FastAPI()

# Directory where models are mapped
MODELS_DIR = "/app/models"
INFO_FILE = os.path.join(MODELS_DIR, "best_model_info.json")
TEMP_DIR = "/tmp"

def get_best_model_path():
    """Reads the JSON info file to find the current best model filename."""
    if not os.path.exists(INFO_FILE):
        return None
    try:
        with open(INFO_FILE, "r") as f:
            info = json.load(f)
        filename = info.get("file")
        return os.path.join(MODELS_DIR, filename) if filename else None
    except Exception:
        return None

def load_model():
    """Safely load the model defined in the info file."""
    model_path = get_best_model_path()
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def align_features(df, model):
    """
    Critical Fix: Ensures the input DataFrame has the EXACT same columns
    as the trained model expects, in the correct order.
    """
    model_features = None

    # 1. Try to detect expected features from the model object
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    elif hasattr(model, "get_booster"): # XGBoost specific
        try:
            model_features = model.get_booster().feature_names
        except:
            pass

    if model_features is None:
        print("Warning: Could not detect model feature names. Skipping alignment.")
        return df

    # 2. Add missing columns (fill with 0)
    missing_cols = set(model_features) - set(df.columns)
    if missing_cols:
        print(f"Debug: Adding {len(missing_cols)} missing columns (filled with 0) to match model.")
        for c in missing_cols:
            df[c] = 0

    # 3. Drop extra columns and Reorder to match model exactly
    # This prevents "ValueError: Feature mismatch"
    return df[model_features]

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Validation
    is_gzip = file.filename.endswith('.gz') or file.content_type == 'application/gzip'
    if not (file.filename.endswith('.csv') or is_gzip):
         raise HTTPException(status_code=400, detail="File must be a CSV or CSV.GZ")

    # 2. Check if Model exists
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready. Train in Airflow first.")

    # 3. Process the file using the SHARED PIPELINE
    raw_temp_path = os.path.join(TEMP_DIR, "temp_raw.csv")
    clean_temp_path = os.path.join(TEMP_DIR, "temp_clean.csv")

    try:
        # A. Write uploaded file to disk
        with open(raw_temp_path, "wb") as buffer:
            if is_gzip:
                content = await file.read()
                buffer.write(gzip.decompress(content))
            else:
                shutil.copyfileobj(file.file, buffer)

        # A.2 Capture LoanIDs (Optimization)
        raw_id_df = pd.DataFrame()
        try:
            cols = pd.read_csv(raw_temp_path, nrows=0).columns
            if 'LoanID' in cols:
                raw_id_df = pd.read_csv(raw_temp_path, usecols=['LoanID'])
            else:
                temp_df = pd.read_csv(raw_temp_path, usecols=[0])
                raw_id_df = pd.DataFrame(index=temp_df.index)
        except Exception as e:
            print(f"Warning: Could not extract LoanIDs upfront: {e}")

        # B. Run the Cleaning Pipeline
        print(f"Running build_dataset on {raw_temp_path}...")
        try:
            build_dataset(raw_temp_path, clean_temp_path)
        except NameError:
             raise HTTPException(status_code=500, detail="Pipeline function 'build_dataset' not found.")
        except Exception as e:
             # [FIX] Log full error to console
             traceback.print_exc()
             raise HTTPException(status_code=400, detail=f"Data Pipeline Error: {str(e)}")

        # C. Read the Cleaned Data
        if not os.path.exists(clean_temp_path):
            raise HTTPException(status_code=500, detail="Pipeline failed to generate clean file.")

        df_clean = pd.read_csv(clean_temp_path)

        # D. PREPARE FOR PREDICTION
        output_ids = None

        # Logic 1: Check if LoanID survived
        if 'LoanID' in df_clean.columns:
            output_ids = df_clean['LoanID']
            df_clean = df_clean.drop(columns=['LoanID'])
        # Logic 2: Map back to raw IDs
        elif not raw_id_df.empty and len(raw_id_df) == len(df_clean):
            if 'LoanID' in raw_id_df.columns:
                output_ids = raw_id_df['LoanID']
            else:
                output_ids = raw_id_df.index
        # Logic 3: Fallback
        else:
            msg = f"Row count changed (Raw: {len(raw_id_df)}, Clean: {len(df_clean)}). Using sequential IDs."
            print(f"Warning: {msg}")
            output_ids = pd.Series(range(len(df_clean)), name='RowIndex_Clean')

        # Drop Target if present (Safety check)
        if 'Default' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Default'])

        # E. Predict (With Alignment Fix)
        try:
            # [FIX] Align features before predicting to prevent XGBoost mismatch
            df_clean = align_features(df_clean, model)
            predictions = model.predict(df_clean)
        except Exception as e:
             # [FIX] Log full error to console
             traceback.print_exc()
             raise HTTPException(status_code=400, detail=f"Model Prediction Error: {str(e)}")

        # F. Prepare Result
        output_df = pd.DataFrame()
        # Handle explicit indices vs Series vs Arrays
        if hasattr(output_ids, 'values'):
            output_df['LoanID'] = output_ids.values
        else:
            output_df['LoanID'] = output_ids

        output_df['predicted_default'] = predictions

        # G. Return CSV
        stream = io.StringIO()
        output_df.to_csv(stream, index=False)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions_result.csv"

        if not raw_id_df.empty and len(raw_id_df) != len(df_clean):
             response.headers["X-Row-Count-Mismatch"] = f"Input: {len(raw_id_df)}, Output: {len(df_clean)}"

        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")
    finally:
        if os.path.exists(raw_temp_path): os.remove(raw_temp_path)
        if os.path.exists(clean_temp_path): os.remove(clean_temp_path)

@app.get("/health")
def health():
    model_path = get_best_model_path()
    return {"status": "ok", "model_ready": model_path is not None}

@app.get("/info")
def model_info():
    if os.path.exists(INFO_FILE):
        with open(INFO_FILE, "r") as f: return json.load(f)
    return {"error": "No model info found"}
