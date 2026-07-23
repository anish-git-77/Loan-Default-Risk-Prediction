# 💳 Loan Default Risk Prediction

An end-to-end credit risk model that predicts the probability a borrower will experience serious delinquency (90+ days past due) within 2 years, built on the ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle dataset.

🔗 **[http://localhost:8501/]

---

## Overview

Banks need to decide loan approvals *before* extending credit, using only application-time information. This project frames that as a binary classification problem — predicting `SeriousDlqin2yrs` (1 = will default, 0 = won't) — and builds a full pipeline from raw, messy data through to a deployed, explainable risk-scoring app.

**Key results:**

| Metric | Score |
|---|---|
| Test ROC-AUC | **0.863** |
| Test KS-Statistic | **56.8** |
| Validation → Test AUC gap | 0.007 (confirms no overfitting) |
| Business-optimal decision threshold | 0.71 (vs. naive 0.5) |

---

## Problem

- 150,000 loan applicants, 10 raw financial features
- Severe class imbalance: **93.3% non-default vs. 6.7% default**
- Goal: rank applicants by risk, and set a decision threshold based on actual business cost, not accuracy

---

## Pipeline

```
Raw data (Kaggle CSV)
   │
   ├─ Data Cleaning
   │    • dropped 1 impossible row (age = 0)
   │    • fixed sentinel/placeholder codes (96, 98) in late-payment columns
   │    • median imputation + missingness flags for MonthlyIncome, NumberOfDependents
   │    • winsorized outliers (RevolvingUtilization, DebtRatio) at 99th percentile
   │
   ├─ Feature Engineering (7 new features)
   │    • TotalPastDue, HasAnyPastDue, DebtToIncomeInteraction,
   │      IncomePerDependent, CreditLinesPerAge, UtilizationTimesLines,
   │      IsRetirementAge
   │
   ├─ Stratified 70/15/15 Train/Val/Test Split
   │
   ├─ Modeling (imbalance handled via class weights, not SMOTE)
   │    • Logistic Regression  → 0.859 val ROC-AUC
   │    • Random Forest        → 0.869 val ROC-AUC
   │    • XGBoost (final)      → 0.870 val ROC-AUC
   │
   ├─ Evaluation
   │    • ROC-AUC, Precision-Recall AUC, KS-statistic
   │    • Business-cost threshold tuning (0.71 optimal vs. 0.5 default)
   │
   ├─ Explainability
   │    • Global feature importance (XGBoost gain)
   │    • SHAP values for individual prediction explanations
   │
   └─ Deployment
        • predict_pipeline.py — reusable inference function
        • app.py — Streamlit web app
```

---

## Repository Structure

```
├── app.py                          # Streamlit web app
├── predict_pipeline.py             # Core inference pipeline (preprocessing + prediction)
├── xgb_model.json                  # Trained XGBoost model (native format, version-stable)
├── rf_model.pkl                    # Trained Random Forest model
├── log_reg_model.pkl               # Trained Logistic Regression model
├── scaler.pkl                      # StandardScaler fit on training data (for Logistic Regression)
├── preprocessing_artifacts.json    # Medians, caps, and threshold used at inference time
├── requirements.txt                # Python dependencies
├── INTERVIEW_PREP_SUMMARY.md       # Full project write-up and Q&A prep
└── README.md
```

---

## Setup & Usage

### 1. Clone and install

```bash
git clone https://github.com/<anish-git-77>/loan-risk-prediction.git
cd loan-risk-prediction
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Run the app locally

```bash
streamlit run app.py
```

### 3. Or use the pipeline directly in Python

```python
from predict_pipeline import predict_risk

applicant = {
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

result = predict_risk(applicant)
print(result)
# {'probability_of_default': 0.14, 'threshold_used': 0.71, 'decision': 'LOW RISK - Approve'}
```

---

## Why These Design Choices

- **Class weights over SMOTE** — avoids data leakage from generating synthetic points before the train/test split.
- **Median imputation + missingness flags** — robust to skewed income/dependents data, while preserving the signal that *missingness itself* can carry.
- **Cost-based threshold tuning** — the default 0.5 cutoff assumes false positives and false negatives cost the same, which is false in lending (a missed default is far costlier than an unnecessary rejection).
- **SHAP explainability** — credit decisions are subject to fair-lending regulations requiring justification for adverse actions; global feature importance alone isn't enough.
- **XGBoost native model format** — `save_model()`/JSON instead of pickle, for version-stable, cross-environment portability.

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Streamlit`

---

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle competition dataset, 150,000 rows, 11 features.

---

## Disclaimer

This is a portfolio/educational project. The cost ratios used for threshold tuning are illustrative, not derived from real loan/interest data. A production deployment would additionally require authentication, audit logging, and drift monitoring.
