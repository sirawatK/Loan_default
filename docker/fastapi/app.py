from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import io
import os

app = FastAPI()

# Path to the model (Mapped from Airflow)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/production_model.pkl")

def load_model():
    """Safely load model, return None if not found"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        # We assume Airflow saved it using joblib or pickle
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Check if file is CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    # 2. Check if Model exists
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready yet. Please train the model in Airflow first.")

    try:
        # 3. Read CSV into Pandas
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 4. Make Prediction
        # Note: Ensure your CSV columns match what the model expects!
        # You might need to select specific columns here, e.g., df[features]
        predictions = model.predict(df)

        # 5. Append Predictions to DataFrame
        df['predicted_result'] = predictions

        # 6. Convert back to CSV for download
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]),
                                     media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "model_exists": os.path.exists(MODEL_PATH)}
