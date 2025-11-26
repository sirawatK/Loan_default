import pandas as pd
import numpy as np
from sklearn.utils import resample
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataPipeline")

try:
    from src.data.feature_engineering import add_features
except ImportError:
    try:
        from data_pipeline import add_features
    except ImportError:
        logger.warning("Feature engineering module not found. Skipping add_features.")
        def add_features(df): return df

def clean_categories(df):
    """Standardizes string formatting and maps binary columns to 0/1."""
    cat_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]
    # Only process columns that actually exist
    existing_cols = [c for c in cat_cols if c in df.columns]
    for c in existing_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    bin_map = {'yes':1, 'no':0, 'true':1, 'false':0}
    for c in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        if c in df.columns:
            df[c] = df[c].map(bin_map)
    return df

def remove_outliers(df):
    """Removes rows based on IQR method. Note: This reduces dataset size."""
    num_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
    ]
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            # Filter rows
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

def build_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # 1. Cleaning & Feature Engineering
    df = clean_categories(df)
    df = remove_outliers(df)
    df = add_features(df)

    # NOTE: We do NOT drop LoanID here globally.
    # We need it to survive until the end for Prediction/Inference mapping.

    # 2. One-Hot Encoding
    encode_cols = ['Education','EmploymentType','MaritalStatus',
                   'LoanPurpose','InterestLevel','DTIBucket']
    # Only encode columns present in the data to avoid KeyErrors
    existing_encode_cols = [c for c in encode_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_encode_cols, drop_first=True)

    # TRAINING VS INFERENCE
    target_col = "Default"

    if target_col in df.columns:
        # === TRAINING MODE ===
        # The target exists, so we prepare the data for model training.
        logger.info("Target column found. Training mode enabled.")

        # A. Drop LoanID: Models should not learn from ID columns (Data Leakage).
        if "LoanID" in df.columns:
            logger.info("Dropping LoanID for training.")
            df = df.drop(columns="LoanID")

        # B. Balance Data: Undersample the majority class to fix imbalance.
        df_minority = df[df[target_col]==1]
        df_majority = df[df[target_col]==0]

        if len(df_minority) > 0 and len(df_majority) > 0:
            df_majority_undersampled = resample(
                df_majority, replace=False, n_samples=len(df_minority), random_state=42
            )
            df = pd.concat([df_majority_undersampled, df_minority])
            logger.info(f"Class distribution after balancing: {df[target_col].value_counts().to_dict()}")
    else:
        # === INFERENCE MODE ===
        # The target is missing. This is likely an API request.
        # 1. Do NOT balance data (we must predict on real distribution).
        # 2. Preserve LoanID (FastAPI needs this to return results to the user).
        logger.info("Target column not found. Inference mode enabled (Preserving LoanID).")

    logger.info(f"Final cleaned dataset shape: {df.shape}")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path}")

if __name__ == "__main__":
    pass
