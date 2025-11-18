import pandas as pd

def load_csv(path: str):
    """Load CSV with error handling."""
    try:
        df = pd.read_csv(path)
        print(f"[OK] Loaded dataset: {path} ({df.shape[0]} rows)")
        return df
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None
