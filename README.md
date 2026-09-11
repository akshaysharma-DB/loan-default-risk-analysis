# Loan Default Risk Analysis

An end-to-end data analytics project using Lending Club loan data to understand default risk and support risk-based lending decisions.

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset & Source](#dataset--source)
- [Tools & Technologies](#tools--technologies)
- [Analytical Approach](#analytical-approach)
- [Key Findings & Insights](#key-findings--insights)
- [Model Performance](#model-performance)
- [Recommended Business Decisions](#recommended-business-decisions)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

## Project Objective

The idea behind this project was to understand **why some borrowers default on their loans and whether that risk can be identified before lending**.

I analyzed historical Lending Club loan data to find patterns in borrower and loan characteristics that are associated with defaults. I then built a Logistic Regression model to estimate the probability of a borrower defaulting.

The final goal was not just to build a model, but to see **how its predictions could be used to make better lending decisions and manage the cost of taking on risky loans.**

## Dataset & Source

This project uses the **Lending Club Loan Data** dataset from Kaggle.

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data)

The downloaded dataset contains two files:

```text
Raw/
├── loan.csv
└── LCDataDictionary.xlsx
```

### `loan.csv`

The main dataset contains approximately:

- **2.26 million loan records**
- **145 columns**
- **~11.6 GB** raw CSV file

### `LCDataDictionary.xlsx`

The dataset is accompanied by a **Loan Data Dictionary**, which provides definitions and descriptions of the columns contained in the loan dataset.

I used the data dictionary as a reference to understand the meaning of the available fields and determine which variables represented information available at the time of lending versus information generated after the loan was issued.

### Data Used for Modeling

The raw dataset was filtered to include only loans with a known final outcome:

```text
Fully Paid    → 0 (Non-default)
Charged Off   → 1 (Default)
```

After filtering, **1,303,607 loans** were retained for analysis and modeling.

## Tools & Technologies

- **Python** — Data analysis, cleaning, feature engineering, and modeling
- **Pandas & NumPy** — Data manipulation and numerical analysis
- **Scikit-learn** — Logistic Regression and model evaluation
- **VS Code** — Development environment
- **Git & GitHub** — Version control and project management

## Analytical Approach

I approached this project step by step, starting with the raw Lending Club data and gradually turning it into a model that could help answer a practical question: **how can we identify and manage loan default risk?**

### Project Workflow

**Raw Data → Data Cleaning → Feature Engineering → Exploratory Data Analysis → Train/Test Split → Logistic Regression → Model Evaluation → Business Decision Analysis**

This was the path I followed from understanding the dataset to evaluating whether the model's predictions could actually support a lending decision.

### 1. Raw Data

I started with around **2.26 million loan records** from the Lending Club dataset. Before working with the data, I went through the accompanying data dictionary to understand what the different fields represented.

### 2. Data Cleaning

The raw dataset had a large number of columns and quite a few missing values. I cleaned the data by removing fields that were not useful for the analysis or had too much missing information.

After cleaning, around **1.3 million loan records** were retained for the analysis.

### 3. Feature Engineering

Once the data was cleaned, I prepared it for analysis and modeling and ended up with **83 features**.

This involved converting variables into usable formats, encoding categorical information, and creating additional features that could help capture differences in borrower and loan risk.

### 4. Exploratory Data Analysis

Before jumping into machine learning, I wanted to understand what the data was actually telling me.

I looked at how default rates changed across factors such as **loan grade, DTI, income, loan purpose, and geographic risk**. This helped me identify the patterns that were worth carrying forward into the modeling stage.

### 5. Train/Test Split & Leakage Prevention

I then split the data into training and testing sets so that I could evaluate the model on data it had not seen before.

One important issue I came across here was **target leakage** with the state-risk feature. Since this feature was based on historical default rates, using information from the entire dataset could allow information from the test set to influence the model.

To avoid that, I calculated the state-risk values using **training data only** and then applied those values to the test data.

### 6. Logistic Regression

I used **Logistic Regression** to estimate the probability that a borrower would default.

Instead of simply saying *“default”* or *“no default,”* the model gives a probability for each borrower. This became important later because it allowed me to test different risk thresholds.

### 7. Model Evaluation

I evaluated the model using **ROC-AUC, precision, recall, F1-score, accuracy, and the confusion matrix**.

But I didn't want to stop at model accuracy. I also looked at how the model behaved when the probability threshold was changed, because the threshold ultimately affects how the lender would act on the predictions.

### 8. Business Decision Analysis

Finally, I connected the model back to the actual lending problem.

Instead of only asking *“How accurate is the model?”*, I asked **“What happens if the business actually uses these predictions to make lending decisions?”**

I compared different probability thresholds and estimated the cost of incorrect decisions. Based on the analysis and the cost assumptions used in the project, the **30% probability threshold** was selected as the starting point for risk screening.

## Key Findings & Insights

### 1. Loan Grade Clearly Separates Risk

One of the strongest patterns I found was in loan grades. The default rate went from **6.08% for Grade A loans to 50.07% for Grade G loans**.

So, a borrower in Grade G was much more likely to default than a borrower in Grade A.

**Why it matters:** This suggests that loan grade can be a useful starting point for deciding how much attention an application needs. Higher-risk grades could be reviewed more carefully rather than being treated the same as lower-risk loans.

---

### 2. High DTI Stands Out as a Risk Signal

I also found a noticeable difference when looking at borrowers' Debt-to-Income (DTI) ratios. The default rate was **14.86% for borrowers with DTI below 10**, compared with **32.20% for borrowers with DTI above 40**.

That's a **17.34 percentage-point difference**.

**Why it matters:** A borrower who is already carrying a high level of debt relative to their income may have less room to take on another loan. DTI could therefore be useful as one of the signals for deciding when an application needs a closer look.

---

### 3. Location Adds Some Extra Information

I didn't want to assume that every state carries the same level of risk, so I created a historical state-risk measure.

The borrowers in the **lowest-risk 20% of states had a 16.96% default rate**, while the **highest-risk 20% had a 22.56% default rate**.

That's a **5.60 percentage-point difference**.

**Why it matters:** Location appears to provide some additional information about risk. However, I would use it as a supporting signal rather than making a lending decision based on location alone.

---

### 4. The Model Can Find Many of the Actual Defaults, but It Isn't Perfect

The Logistic Regression model achieved an **ROC-AUC of 0.7157** and was able to identify **64% of the borrowers who eventually defaulted**.

At the same time, its precision for the default class was **33%**. In other words, when the model flags someone as likely to default, a significant number of those borrowers would still go on to repay their loans.

**Why it matters:** I wouldn't use this model to automatically reject borrowers. I would use it to **prioritize applications for additional review**, allowing the lending team to spend more time on applications that appear riskier.

> **ROC-AUC in simple terms:** It tells us how well the model can rank borrowers from lower risk to higher risk across different probability cutoffs. **0.5** would mean the model is essentially guessing, while **1.0** would mean perfect separation. The model's **0.7157** shows that it has useful ability to distinguish between the two groups.

---

### 5. A 50% Cutoff Isn't Necessarily the Best Business Decision

The model gives each borrower a probability of default. Instead of automatically saying *"above 50% means high risk,"* I tested different cutoffs and looked at the estimated cost of getting the decision wrong.

| Default Probability Cutoff | Estimated Cost |
|---|---:|
| 50% | **$438.64M** |
| 30% | **$370.18M** |

Under the assumptions used in this project, moving the cutoff from **50% to 30% reduced the estimated cost by $68.46M (15.61%)**.

**Why it matters:** The model's job shouldn't end with a prediction. The business still needs to decide **how much risk it is willing to accept**. A lower cutoff catches more potential defaults, but it can also flag more borrowers who would have repaid. The right balance depends on the lender's actual costs and risk appetite.

> **Note:** These dollar figures are estimates based on the cost assumptions used in this analysis. They are meant to compare different decision thresholds, not represent guaranteed savings.

## Model Performance

The Logistic Regression model was evaluated on its ability to identify borrowers who were likely to default.

| Metric | Result |
|---|---:|
| **ROC-AUC** | **0.7157** |
| **Accuracy** | **66.33%** |
| **Recall (Default)** | **64%** |
| **Precision (Default)** | **33%** |

The model was able to identify **64% of borrowers who actually defaulted**, with an ROC-AUC of **0.7157**.

I also tested different probability thresholds to see how changing the definition of a **high-risk borrower** would affect the estimated cost of lending decisions.

| Default Probability Threshold | Estimated Cost |
|---|---:|
| **50%** | **$438.64M** |
| **30%** | **$370.18M** |

The **30% threshold produced an estimated cost that was $68.46M lower** than the 50% threshold under the assumptions used in this project.

This showed that the model's output can be used beyond simply predicting default—it can also help evaluate **different lending strategies and their potential financial impact**.

## Recommended Business Decisions

Based on the analysis, I would use the model as a **risk-screening tool rather than an automatic loan approval or rejection system**.

### 1. Prioritize Higher-Risk Applications

The model's predicted default probability can help identify applications that may need additional review.

Applications with higher predicted risk could be reviewed more carefully, while lower-risk applications could continue through the standard process.

### 2. Give More Attention to High-Risk Borrower Profiles

The analysis showed clear differences in default rates across factors such as **loan grade and DTI**.

These factors could be used alongside the model's prediction to understand *why* an application is being flagged as higher risk.

### 3. Use the 30% Threshold as a Starting Point

The analysis showed that the **30% probability threshold had a lower estimated cost than the 50% threshold**, with an estimated difference of **$68.46M** under the project's assumptions.

I would therefore use **30% as the starting point for risk screening**, rather than treating 50% as the default cutoff.

### 4. Keep Human Review in the Decision

The model is useful for identifying potential risk, but it is not perfect. Since some borrowers flagged as high risk would still repay their loans, the prediction should support the lending team's decision rather than completely replace it.

**Overall, the goal is to use the model to make lending decisions more risk-aware, while still allowing the business to apply its own policies and risk appetite.**

## Limitations & Future Improvements

There are a few areas where I would take the project further:

- **Model improvement:** Test other classification models and compare their performance with Logistic Regression.
- **Feature improvement:** Explore additional features and interactions that could provide more information about borrower risk.
- **Threshold optimization:** Recalculate the optimal threshold using real lending costs and the lender's actual risk tolerance.
- **Model interpretability:** Add clearer explanations of which factors are contributing most to an individual borrower's predicted risk.
- **Dashboard:** Build a Power BI dashboard to make the findings and risk analysis easier for non-technical stakeholders to explore.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/akshaysharma-DB/loan-default-risk-analysis.git
cd loan-default-risk-analysis
```

### 2. Install the required libraries

```bash
pip install pandas numpy scikit-learn
```

### 3. Add the dataset

Download the **[Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data)** dataset and place the required raw files in the `Raw/` directory:

```text
Raw/
├── loan.csv
└── LCDataDictionary.xlsx
```

### 4. Run the analysis

Run the scripts from the `src/` directory in this order:

```text
Loan_analysis.py
      ↓
loan_features.py
      ↓
Loan_Split.py
      ↓
Loan_eda.py
      ↓
Loan_model_baseline.py
      ↓
Loan_model_v2.py
```

## Conclusion

This project started with a simple question: **can historical loan data help identify borrowers who are more likely to default?**

Through data cleaning, exploratory analysis, feature engineering, and Logistic Regression, I was able to identify several patterns in default risk and build a model that provides a probability of default for each borrower.

The analysis also showed that building a model is only part of the problem. **How those predictions are used matters just as much.** By testing different risk thresholds and estimating their potential cost, I was able to connect the model's predictions to a practical lending decision.

Overall, the project helped me understand how a data analysis workflow can move from **raw data → insights → prediction → business decision**.
