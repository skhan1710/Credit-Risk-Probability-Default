# Credit Risk PD Scorecard

A credit risk modeling project built on 1.25M LendingClub personal loans. Trains a logistic regression scorecard and XGBoost challenger model to estimate probability of default (PD) at the loan application stage, with validation framed against SR 11-7 supervisory guidance.


## Dataset

| Detail | Value |
|--------|-------|
| Source | LendingClub Accepted Loans 2007–2018 |
| Filtered sample | 1,263,259 loans |
| Period | 2012–2017 |
| Class split | 78.8% good · 21.2% bad |
| Train / Test split | 2012–2015 development · 2016–2017 out-of-time |

2012–2017 was chosen because every loan in that window had enough time to reach a terminal status by the dataset's 2018 cutoff. Loans still marked Current, In Grace Period, or Late 16-30 days at cutoff were dropped — their outcome is unknown, so they can't be used as training examples.

LendingClub's proprietary risk grades and interest rates were excluded entirely. Both variables are assigned by LendingClub after their own internal credit assessment, so including them would mean partially copying their model rather than building an independent one.


## How default is defined

Each loan was labeled using terminal loan_status:

| Status | Label |
|--------|-------|
| Fully Paid | 0 (good) |
| Does not meet credit policy — Fully Paid | 0 (good) |
| Charged Off | 1 (default) |
| Default | 1 (default) |
| Late (31-120 days) | 1 (default) |
| Does not meet credit policy — Charged Off | 1 (default) |

Late 31-120 days is included as default because at that stage recovery is unlikely and the loss is effectively realized. Late 16-30 days is dropped — too early to call.


## Methodology

### Feature Selection — Weight of Evidence / Information Value

Before modeling, every candidate feature was scored by Information Value (IV) to determine whether it carries enough signal to be worth including.

The IV formula for one variable across all its bins:

```
IV = Σ (% Goods in bin - % Bads in bin) × ln(% Goods in bin / % Bads in bin)
```

The difference term `(% Goods - % Bads)` acts as a weight — it scales each bin's contribution by how much of the population it represents. The log ratio `ln(% Goods / % Bads)` captures the relative imbalance between good and bad borrowers in that bin. Multiplying them means a bin only contributes significantly when both conditions are true: it covers a meaningful share of the portfolio AND shows a strong separation between goods and bads.

A variable with strong signal in only a tiny fraction of loans scores low IV regardless of how extreme that signal is. This is intentional.

| Threshold | Interpretation |
|-----------|---------------|
| IV < 0.02 | Useless — drop |
| 0.02 – 0.10 | Weak |
| 0.10 – 0.30 | Medium |
| > 0.30 | Strong |

Variables below IV = 0.02 were dropped unless domain knowledge justified keeping them (delinquency history, public records — standard regulatory scorecard variables).

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

No strong predictors (IV > 0.3) appear because grade and sub_grade were excluded. The Gini range of 0.37–0.41 is consistent with published benchmarks on this dataset using application-time features only.

### Champion Model — Logistic Regression Scorecard

Before fitting, all features were WoE-transformed. Each raw value is replaced by its bin's WoE value:

```
WoE for bin = ln(% Goods in bin / % Bads in bin)
```

This puts every feature on the same log-odds scale regardless of its original units, handles missing values naturally as their own bin, and produces a perfectly linear relationship between inputs and the model's log-odds output. The combination of WoE transformation and logistic regression has been the industry standard in credit scoring for decades because the math lines up cleanly and every coefficient is directly interpretable.

The model estimates:

```
log odds of default = b0 + b1×term_woe + b2×fico_woe + b3×dti_woe + ...

PD = 1 / (1 + e^-(log odds))
```

Coefficients are found by minimizing log loss across all training loans via gradient descent.

### Challenger Model — XGBoost

XGBoost was trained on raw features without WoE transformation. Tree-based models find their own splits internally, so the scaling problem WoE solves for logistic regression doesn't apply. Categorical columns were encoded as native category dtype. Class imbalance (78.8% / 21.2%) was handled via scale_pos_weight = 3.72, which inflates the gradient penalty for misclassified bad loans during training.

XGBoost builds 500 trees sequentially. Each tree predicts the residual errors of all previous trees combined, correcting mistakes incrementally. The final prediction is the sum of all 500 corrections converted to a probability via sigmoid.


## Results

| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| AUC | 0.6842 | 0.7051 |
| Gini | 0.3685 | 0.4102 |
| KS | 0.2625 | 0.2969 |
| PSI (train vs OOT) | 0.0030 | 0.0012 |

AUC answers: if you randomly pick one defaulted loan and one good loan, what is the probability the model ranked the defaulted loan as riskier? At 0.705, XGBoost gets this right 70.5% of the time. Gini is AUC rescaled to start at 0 (random) rather than 0.5. KS measures the maximum separation between the cumulative default and non-default score distributions — at the optimal threshold, XGBoost separates the two groups 4.4 percentage points better than logistic regression.

XGBoost outperforms on all discrimination metrics. The gap is modest because both models are working from the same raw application-time features without LendingClub's proprietary risk grades. This is the honest ceiling of the available signal.


## Calibration

Discrimination and calibration measure different things. Gini and KS measure rank ordering — does the model correctly identify which borrowers are riskier than others? Calibration measures absolute accuracy — if the model says 20% PD, do 20% of those borrowers actually default?

LR and XGBoost have opposite calibration errors.

LR underestimates risk at higher PD buckets. The model says 25% but 33% actually default. This is a known property of logistic regression — it tends to pull predictions toward the mean.

XGBoost overestimates risk across the board. The model says 35% but only 20% actually default. The cause is scale_pos_weight = 3.72. During training, every misclassified bad loan carries a 3.72x heavier gradient penalty than a misclassified good loan. The model compensates by pushing predicted probabilities higher than they should be to avoid the penalty — which systematically inflates PD estimates.

This matters for loss forecasting. If the model is used to calculate CECL reserves, miscalibrated PDs produce incorrect reserve estimates regardless of how good the Gini is. A production deployment would apply Platt scaling to correct XGBoost's calibration before using predicted PDs for any dollar-value calculation.


## SHAP Feature Importance


SHAP assigns each feature a value for each individual prediction i.e. how much did this feature push this borrower's PD above or below the portfolio average?

| Feature | Mean \|SHAP\| |
|---------|--------------:|
| term | 0.513 |
| installment | 0.499 |
| loan_amnt | 0.330 |
| fico_avg | 0.213 |
| acc_open_past_24mths | 0.193 |

IV and SHAP rankings diverge. IV ranked fico_avg second (IV = 0.116). SHAP ranks installment second (mean SHAP = 0.499). IV measures each variable's signal in isolation — how well does fico_avg alone separate goods from bads? SHAP measures actual influence inside the trained model, which includes interactions between features. Installment and loan_amnt are correlated with term (longer term loans have lower monthly payments on the same principal), and the model learned to use all three together. IV can't see that; SHAP can.


## Population Stability

PSI measures whether the distribution of predicted scores shifted between the training period (2012–2015) and the out-of-time period (2016–2017):

```
PSI = Σ (% loans in bin, OOT - % loans in bin, train) × ln(% OOT / % train)
```

The difference term captures how much each score bucket shifted. The log ratio captures how severe that shift was relative to the original distribution. Both models returned PSI well below the 0.10 stability threshold, confirming the borrower population seen in 2016–2017 was not meaningfully different from the development sample. The model was not asked to score a population it had never seen.


## Limitations

| Limitation | Root cause | What would fix it |
|------------|------------|-------------------|
| Gini ceiling ~0.41 | grade and sub_grade excluded — they encode LendingClub's full credit assessment in a single variable | Including them pushes Gini to ~0.45 but creates dependence on a third-party risk model |
| XGBoost miscalibration | scale_pos_weight inflates bad loan gradients, pushing predicted PDs above actual default rates | Platt scaling or isotonic regression post-training |
| No drift monitoring | PSI computed once on a fixed holdout — not a live monitoring pipeline | Rolling PSI on production score distributions |
| Single origination environment | 2012–2017 was a stable post-GFC credit cycle | Retesting on a period that includes a recession would stress-test the model |
| Static feature set | No macroeconomic overlays (unemployment, GDP growth) | Adding FRED macro variables as time-varying features |


## Scoring Demo

A Streamlit app (`app.py`) takes borrower inputs and returns a predicted PD in real time. It loads the serialized XGBoost model, encodes categorical features to match the training dtype, assembles a single-row dataframe, and calls predict_proba.

```bash
streamlit run app.py
```


## Repository Structure

```
├── lendingclub.ipynb    # Full pipeline: data → features → models → validation
├── app.py               # Streamlit scoring demo
├── xgb_model.pkl        # Serialized XGBoost model
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost optbinning shap streamlit joblib

# Run notebook top to bottom — all outputs are self-contained
# Launch scoring demo
streamlit run app.py
```
