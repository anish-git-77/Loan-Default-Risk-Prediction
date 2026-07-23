"""
Loan Default Risk Prediction Pipeline
======================================
Takes raw applicant data (same schema as the original Kaggle dataset)
and returns a risk decision, using the exact same preprocessing and
feature engineering logic used during training -- avoiding train/serve skew.
"""

import json
import numpy as np
import pandas as pd
import joblib

# ---- Load trained artifacts once (model, scaler, preprocessing constants) ----
MODEL = joblib.load('xgb_model.pkl')
with open('preprocessing_artifacts.json') as f:
    ARTIFACTS = json.load(f)

RAW_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age',
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates every cleaning + feature engineering step from training."""
    df = df.copy()

    # --- enforce numeric dtypes first (raw input may arrive as None/object) ---
    numeric_cols = [c for c in RAW_COLUMNS if c not in ('MonthlyIncome', 'NumberOfDependents')]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['MonthlyIncome'] = pd.to_numeric(df['MonthlyIncome'], errors='coerce')
    df['NumberOfDependents'] = pd.to_numeric(df['NumberOfDependents'], errors='coerce')

    # --- missingness flags (must happen BEFORE imputation) ---
    df['MonthlyIncome_was_missing'] = df['MonthlyIncome'].isnull().astype(int)
    df['NumberOfDependents_was_missing'] = df['NumberOfDependents'].isnull().astype(int)

    # --- median imputation using TRAINING medians (never recompute on new data) ---
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(ARTIFACTS['income_median'])
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(ARTIFACTS['dependents_median'])

    # --- sentinel code correction (96/98 -> training-time real max) ---
    for col, cap in ARTIFACTS['sentinel_caps'].items():
        df[col] = df[col].replace({96: cap, 98: cap})

    # --- outlier capping using TRAINING quantiles ---
    df['RevolvingUtilizationOfUnsecuredLines'] = np.minimum(
        df['RevolvingUtilizationOfUnsecuredLines'], ARTIFACTS['utilization_cap'])
    df['DebtRatio'] = np.minimum(df['DebtRatio'], ARTIFACTS['debt_ratio_cap'])

    # --- feature engineering (identical to training) ---
    df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] +
                           df['NumberOfTime60-89DaysPastDueNotWorse'] +
                           df['NumberOfTimes90DaysLate'])
    df['HasAnyPastDue'] = (df['TotalPastDue'] > 0).astype(int)
    df['DebtToIncomeInteraction'] = df['DebtRatio'] * df['MonthlyIncome']
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['CreditLinesPerAge'] = df['NumberOfOpenCreditLinesAndLoans'] / df['age']
    df['UtilizationTimesLines'] = (df['RevolvingUtilizationOfUnsecuredLines'] *
                                    df['NumberOfOpenCreditLinesAndLoans'])
    df['IsRetirementAge'] = (df['age'] >= 60).astype(int)

    return df


def predict_risk(applicant: dict) -> dict:
    """
    Takes a single applicant's raw data as a dict and returns a risk decision.

    Example input:
    {
        'RevolvingUtilizationOfUnsecuredLines': 0.4,
        'age': 35,
        'NumberOfTime30-59DaysPastDueNotWorse': 1,
        'DebtRatio': 0.3,
        'MonthlyIncome': 5000,
        'NumberOfOpenCreditLinesAndLoans': 5,
        'NumberOfTimes90DaysLate': 0,
        'NumberRealEstateLoansOrLines': 1,
        'NumberOfTime60-89DaysPastDueNotWorse': 0,
        'NumberOfDependents': 2
    }
    """
    df = pd.DataFrame([applicant])[RAW_COLUMNS]
    df = preprocess(df)

    # Match exact column order the model was trained on
    feature_cols = MODEL.get_booster().feature_names
    X = df[feature_cols]

    probability = MODEL.predict_proba(X)[0, 1]
    threshold = ARTIFACTS['best_threshold']
    decision = 'HIGH RISK - Flag for review' if probability >= threshold else 'LOW RISK - Approve'

    return {
        'probability_of_default': round(float(probability), 4),
        'threshold_used': threshold,
        'decision': decision
    }


if __name__ == '__main__':
    # Quick smoke test with a sample applicant
    sample_applicant = {
        'RevolvingUtilizationOfUnsecuredLines': 0.8,
        'age': 29,
        'NumberOfTime30-59DaysPastDueNotWorse': 3,
        'DebtRatio': 0.6,
        'MonthlyIncome': 3000,
        'NumberOfOpenCreditLinesAndLoans': 8,
        'NumberOfTimes90DaysLate': 2,
        'NumberRealEstateLoansOrLines': 0,
        'NumberOfTime60-89DaysPastDueNotWorse': 1,
        'NumberOfDependents': 3
    }
    result = predict_risk(sample_applicant)
    print("Sample prediction:", result)