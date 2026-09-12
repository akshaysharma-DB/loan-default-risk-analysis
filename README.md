# Loan Default Risk Analyzer

An end-to-end machine learning project that estimates loan default probability and turns the prediction into a practical risk-screening workflow for lenders.

The project goes beyond model training: it combines borrower application data, credit-profile information, probability-based risk assessment, threshold analysis, and a Streamlit application that shows applicant-specific risk factors.

## Project Objective

The core business question is:

> **Can historical loan data be used to estimate default risk early enough to support better lending decisions?**

The goal is not to automatically approve or reject borrowers. Instead, the model is designed as a **decision-support and risk-screening tool** that can help a lender identify applications that deserve additional review.

## Dataset

The project uses the **Lending Club Loan Data** dataset from Kaggle.

Source: [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data)

The original dataset contains approximately **2.26 million loan records and 145 columns**. After filtering to loans with a known final outcome (`Fully Paid` or `Charged Off`), approximately **1.30 million records** were retained for analysis and modeling.

The target was defined as:

- `Fully Paid` → `0` (non-default)
- `Charged Off` → `1` (default)

## Business Approach

The project follows this workflow:

```text
Raw Lending Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Analysis
       ↓
Train/Test Split
       ↓
Logistic Regression
       ↓
Probability Prediction
       ↓
Threshold / Cost Analysis
       ↓
Risk Screening Application
```

## Features

The final application model uses information that can reasonably be available when assessing a loan application, including:

### Loan & Applicant Information

- Loan amount
- Funded amount
- Loan term
- Interest rate
- Annual income
- Employment length
- Debt-to-income ratio
- Monthly installment
- Loan purpose
- Home ownership
- Credit history length

### Credit Profile

- Recent delinquencies
- Recent credit inquiries
- Open credit accounts
- Total credit accounts
- Revolving balance
- Revolving utilization
- Mortgage accounts
- Total current balance
- Public records

## Machine Learning Model

The final application model uses **Logistic Regression**.

Logistic Regression was selected because the project needs both:

1. A probability of default rather than only a binary prediction.
2. An interpretable way to understand which applicant characteristics are associated with higher or lower predicted risk.

Feature scaling and categorical encoding are saved with the model so that the Streamlit application applies the same preprocessing used during training.

## Model Performance

The current application model achieved approximately:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7029** |
| Accuracy | **66.07%** |
| Default Recall | **62%** |
| Default Precision | **32%** |

The model is not intended to be a perfect classifier. Its purpose is to provide a useful ranking and probability estimate that can support a lender's review process.

## Why a 30% Threshold?

A probability of 50% is not automatically the best business decision threshold.

The project tested different probability cutoffs and estimated the cost of false negatives and false positives under the project's assumptions.

The earlier threshold analysis showed:

| Threshold | Estimated Total Cost |
|---|---:|
| 50% | **$438.64M** |
| 30% | **$370.18M** |

The 30% threshold therefore produced a lower estimated cost under those assumptions and was selected as the application's starting point for **further review**.

This threshold is not presented as a universal lending rule. A real lender would calibrate it using its own default costs, approval strategy, and risk appetite.

## Streamlit Application

The project includes an interactive Streamlit application where a user can enter a borrower profile and receive:

- Estimated default probability
- Risk level
- Threshold-based recommendation
- Applicant-specific risk factors
- Factors reducing estimated risk
- A summary of the application

The application is designed to answer two questions:

> **How risky does this application look?**

and

> **Which characteristics are driving that assessment?**

The risk-factor section uses the trained model's feature contributions for the individual applicant rather than simply displaying a generic list of important model features.

## Key Analytical Findings

### Loan Grade and Default Risk

Historical analysis showed a strong relationship between loan grade and default rate, with default rates increasing substantially across lower grades.

### Debt-to-Income Ratio

Default rates were also meaningfully higher among borrowers with high DTI. This supports DTI as an important signal when assessing repayment capacity.

### Location Risk

A historical state-risk feature provided additional predictive information. It was calculated using training data only to reduce the risk of target leakage.

### Model Interpretation

The model provides coefficients that can be converted into applicant-specific contributions. This allows the application to identify factors associated with a particular prediction instead of treating the model as a black box.

## Important Modeling Decision: Avoiding Target Leakage

One of the project's important preprocessing decisions was handling the historical state-risk feature.

Because state risk was derived from historical default behavior, calculating it using the complete dataset could allow information from the test set to influence the training process.

The feature was therefore calculated from **training data only** and then applied to the test data.

## Business Use Case

A lender could use a system like this as an additional screening layer:

```text
Loan Application
       ↓
Risk Model
       ↓
Default Probability
       ↓
Risk Threshold
       ↓
Standard Review / Further Review
       ↓
Human Underwriting Decision
```

The model should support underwriting rather than replace it. A prediction is an estimate of risk, not proof that a borrower will or will not default.

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Git & GitHub**

## Project Structure

```text
loan-default-risk-analysis/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── Loan_analysis.py
│   ├── loan_features.py
│   ├── Loan_eda.py
│   ├── Loan_Split.py
│   ├── Loan_model_baseline.py
│   ├── Loan_model_v2.py
│   ├── Loan_model_v3.py
│   ├── predict.py
│   └── Paths.py
│
└── models/
    ├── logistic_model_v3.pkl
    ├── scaler_v3.pkl
    ├── encoder_v3.pkl
    └── features_v3.pkl
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/akshaysharma-DB/loan-default-risk-analysis.git
cd loan-default-risk-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open locally in your browser.

## Limitations

- The model is trained on historical Lending Club data and may not generalize to every lending environment.
- The 30% threshold depends on the cost assumptions used in this project.
- Logistic Regression provides useful interpretability but may not capture every nonlinear relationship in borrower behavior.
- Model associations should not be interpreted as proof of causation.
- The application is a decision-support tool and should not be the sole basis for approving or rejecting a loan.

## Future Improvements

Potential next steps include:

- Compare Logistic Regression with tree-based models.
- Calibrate predicted probabilities.
- Perform stronger cross-validation and hyperparameter tuning.
- Test the model on a time-based holdout to better simulate future lending decisions.
- Improve fairness and bias evaluation across relevant borrower groups.
- Deploy the Streamlit application publicly.
- Add monitoring for model drift and changing default patterns.

## Author

**Akshay Sharma**

Data Analyst | Python | SQL | Power BI | Machine Learning
