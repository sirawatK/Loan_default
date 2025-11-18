import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['LoanToIncome'] = df['LoanAmount'] / df['Income']
    df['EmploymentYears'] = df['MonthsEmployed'] / 12
    df['CreditRisk'] = (df['CreditScore'] < 600).astype(int)

    df['InterestLevel'] = pd.qcut(
        df['InterestRate'], q=3, labels=['low', 'medium', 'high']
    )

    df['DTIBucket'] = pd.cut(
        df['DTIRatio'],
        bins=[0, 15, 30, 50, 200],
        labels=['low', 'medium', 'high', 'extreme']
    )

    return df
