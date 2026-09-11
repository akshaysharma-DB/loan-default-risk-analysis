# Loan Default Risk Analysis

An end-to-end data analytics project using Lending Club loan data to understand default risk and support risk-based lending decisions.

## Project Status

**In development** — the project is being built step by step, from raw data preparation to business-focused insights and Power BI visualization.

### Completed

- Processed **2.26M+ loan records** and performed data quality checks
- Cleaned the dataset and removed post-loan variables to avoid target leakage
- Reduced the working dataset to **1.3M loans and 83 features**
- Engineered borrower, loan, credit-history, and state-risk features
- Performed EDA across loan grade, DTI, income, loan purpose, and state risk
- Created a stratified train/test split and corrected target leakage in state-risk encoding
- Built a Logistic Regression baseline and improved model
- Achieved **71.57% ROC-AUC** and **64% recall for defaults** with the improved model
- Performed cost-based threshold analysis; a **0.30 threshold** reduced estimated cost from **$438.6M to $370.2M (15.6%)** compared with the standard 0.50 threshold

### Currently Working On

- Refining the business interpretation of the model results
- Turning the analysis into clear, decision-focused insights
- Preparing the data and metrics for visualization

### Planned

- Build an interactive **Power BI dashboard**
- Present key default-risk patterns and business insights
- Develop practical, risk-based lending recommendations
- Document the complete analytical workflow and final findings

## Key Findings

- Default rates increased from **6.1% for Grade A** loans to **50.1% for Grade G** loans
- Default rates increased from **14.9% for DTI below 10** to **32.2% for DTI above 40**
- The lowest-income group had a **23.7%** default rate compared with **15.8%** for the highest-income group
- The model's cost analysis showed that the standard 0.50 probability threshold was not the lowest-cost decision point under the project's assumptions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Power BI

## Project Structure

```text
loan-default-risk-analysis/
├── src/
│   ├── Loan_analysis.py
│   ├── loan_features.py
│   ├── Loan_eda.py
│   ├── Loan_Split.py
│   ├── Loan_model_baseline.py
│   ├── Loan_model_v2.py
│   └── Paths.py
├── Cleaned/
├── .gitignore
└── README.md
```

## Objective

To analyze loan data, identify factors associated with default risk, build a model that can estimate default probability, and translate the results into business-focused insights that can support better lending decisions.
