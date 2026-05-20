# Credit Risk PD Scorecard

A credit risk modeling project built on 1.25M LendingClub personal loans (2012–2017). Implements a logistic regression scorecard and XGBoost challenger model following standard industry practices for PD estimation, with model validation framed against SR 11-7 supervisory guidance.

## Dataset

| Detail | Value |
|--------|-------|
| Source | LendingClub Accepted Loans 2007–2018 |
| Filtered sample | 1,263,259 loans · 2012–2017 |
| Target variable | Default (Charged Off, Late 31-120 days) = 1 |
| Class split | 78.8% good · 21.2% bad |
| Train / Test split | 2012–2015 (development) · 2016–2017 (out-of-time) |

LendingClub's proprietary risk grades and interest rates were excluded to build an independent scorecard from raw application-time features only.

## Methodology

### Feature Selection — WoE / IV

All candidate features were binned using optimal binning and scored by Information Value. Features below IV = 0.02 were dropped unless retained on domain knowledge grounds (delinquency history, public records).

| Feature | IV | Category |
|---------|-----|----------|
| term | 0.185 | Medium |
| fico_avg | 0.116 | Medium |
| dti | 0.072 | Weak |
| acc_open_past_24mths | 0.067 | Weak |
| verification_status | 0.052 | Weak |
| mort_acc | 0.042 | Weak |
| loan_amnt | 0.036 | Weak |
| installment | 0.030 | Weak |
| home_ownership | 0.030 | Weak |
| annual_inc | 0.027 | Weak |

No strong predictors (IV > 0.3) appear because grade and sub_grade — which encode LendingClub's own risk model — were excluded. The 0.37–0.41 Gini range is consistent with published benchmarks on this dataset using application-time features only.

### Champion Model — Logistic Regression Scorecard

Raw features were WoE-transformed before fitting. WoE standardizes all inputs onto a log-odds scale, handles missing values as a separate bin, and produces a perfectly linear relationship with the log-odds output — which is why WoE + logistic regression has been the industry standard in credit scoring for decades.

### Challenger Model — XGBoost

Trained on raw features directly (no WoE needed). Categorical columns encoded as native category dtype. Class imbalance handled via scale_pos_weight = 3.72.

## Results

| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| AUC | 0.6842 | 0.7051 |
| Gini | 0.3685 | 0.4102 |
| KS | 0.2625 | 0.2969 |
| PSI | 0.0030 | 0.0012 |

XGBoost outperforms on discrimination metrics. Logistic regression shows better probability calibration — predicted PDs track actual default rates more closely, which matters for loss forecasting and CECL reserve calculations.

Both models show PSI well below 0.10, confirming no meaningful population drift between the development and out-of-time periods.

### Calibration Finding

LR and XGBoost have opposite calibration errors. LR underestimates risk at higher PD buckets (predicted 25%, actual 33%). XGBoost overestimates risk across the board (predicted 35%, actual 20%) — a direct consequence of scale_pos_weight inflating gradients for bad loans during training. A production deployment would apply Platt scaling to correct XGBoost's calibration before using predicted PDs for loss forecasting.

### SHAP Feature Importance

| Feature | Mean SHAP |
|---------|-----------|
| term | 0.513 |
| installment | 0.499 |
| loan_amnt | 0.330 |
| fico_avg | 0.213 |
| acc_open_past_24mths | 0.193 |

SHAP and IV rankings diverge slightly — installment and loan_amnt rank higher in SHAP than IV because SHAP captures interaction effects the model learned, while IV measures each variable's signal in isolation.

## Scoring Demo
 
A Streamlit app (`app.py`) lets you input borrower details and get a predicted PD in real time. It loads the trained XGBoost model and handles all preprocessing — categorical encoding, column ordering — before calling predict_proba.
 
To run locally:
 
```bash
streamlit run app.py
```
 
## Repository Structure
 
```
├── lendingclub.ipynb    # Full pipeline: data → features → models → validation
├── app.py               # Streamlit scoring demo
├── xgb_model.pkl        # Trained XGBoost model
└── README.md
```
 
## How to Run
 
```bash
pip install pandas numpy scikit-learn xgboost optbinning shap streamlit joblib
 
# Run notebook top to bottom — all outputs are self-contained
# Launch scoring demo
streamlit run app.py
```

## Limitations

| Limitation | Detail |
|------------|--------|
| No strong predictors | Gini ceiling ~0.41 without LendingClub's proprietary grade variable |
| Calibration drift | XGBoost overestimates risk due to scale_pos_weight — Platt scaling not yet applied |
| Static model | No drift monitoring or retraining pipeline implemented |
| Single portfolio | Results specific to LendingClub's 2012–2017 origination environment |
