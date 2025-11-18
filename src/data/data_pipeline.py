import pandas as pd
import numpy as np
try:
    
    from src.data.feature_engineering import add_features
except ModuleNotFoundError:
    
    from data_pipeline import add_features

from src.utils.logger import get_logger

logger = get_logger("DataPipeline")

def clean_categories(df):
    cat_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    bin_map = {'yes':1, 'no':0, 'true':1, 'false':0}
    for c in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        df[c] = df[c].map(bin_map)

    return df


def remove_outliers(df):
    num_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
    ]
    
    for col in num_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        after = len(df)

        logger.info(f"{col}: removed {before - after} outliers")
    return df


def build_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    df = clean_categories(df)
    df = remove_outliers(df)
    df = add_features(df)

    df = pd.get_dummies(df,
                        columns=['Education','EmploymentType','MaritalStatus',
                                 'LoanPurpose','InterestLevel','DTIBucket'],
                        drop_first=True)

    logger.info(f"Final cleaned dataset shape: {df.shape}")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    build_dataset("data/raw/Loan_default.csv", "data/processed/clean_Loan_default.csv")
