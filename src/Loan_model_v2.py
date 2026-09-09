import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Load the split data saved earlier ---
X_train = pd.read_csv(r'D:\Downloads\Data\X_train.csv')
X_test = pd.read_csv(r'D:\Downloads\Data\X_test.csv')
y_train = pd.read_csv(r'D:\Downloads\Data\y_train.csv').squeeze()
y_test = pd.read_csv(r'D:\Downloads\Data\y_test.csv').squeeze()

print(X_train.shape, X_test.shape)

# --- Feature scaling ---
# Logistic regression's optimizer struggles when features are on very
# different scales (loan_amnt in thousands vs dti in single digits vs
# 0/1 dummy columns). StandardScaler transforms each column to mean=0,
# std=1, which fixes the convergence warning and helps the model treat
# each feature fairly during optimization.
#
# IMPORTANT: fit the scaler on X_train ONLY, then use that same fitted
# scaler to transform X_test. If we fit on the full dataset (or on test
# too), information about the test set's distribution leaks into
# training -- the same leakage principle as addr_state_risk earlier.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic regression with class weighting ---
# class_weight='balanced' tells the model to penalize mistakes on the
# minority class (default=1) more heavily, roughly in proportion to
# how underrepresented it is (~1:4 here). Without this, the model has
# little incentive to bother predicting class 1 at all, since ignoring
# it barely hurts overall accuracy -- exactly what we saw in the baseline.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# --- Predictions ---
y_pred = model.predict(X_test_scaled)

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

# X_train is still the unscaled DataFrame, so its columns match
# the order features were fed into the model
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
})

# Sort by absolute value -- a large negative coefficient (pushes toward
# "no default") matters just as much as a large positive one (pushes
# toward "default")
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)

print(coefficients.head(15).to_string(index=False))

from sklearn.metrics import roc_auc_score, roc_curve

# --- ROC-AUC ---
# predict_proba gives the actual probability of default (0 to 1) for each
# loan, not just the final yes/no prediction. We need this for AUC, since
# AUC measures how well the model RANKS risk across all possible
# thresholds -- not just the one (0.5) that .predict() used by default.
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 (default)

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")
#ROC-AUC is  0.7157,
# above the 0.70 mark is better for real-world lending risk models

import numpy as np

# --- Cost-based threshold analysis ---
# Instead of the default 0.5 cutoff, we test a range of thresholds and
# calculate the actual dollar cost of each one's mistakes, using each
# loan's real amount rather than one averaged number.

# False negative cost: loan_amnt itself -- money lent out and never
# recovered when the model wrongly predicted "safe" on an actual defaulter.
fn_cost_per_loan = X_test['loan_amnt']

# False positive cost: lost interest income -- what the lender would have
# earned had they approved this loan and it been repaid as expected.
# Simplified to one year of interest; refine with term later if needed.
fp_cost_per_loan = X_test['loan_amnt'] * (X_test['int_rate'] / 100)

results = []
thresholds = np.arange(0.1, 0.95, 0.05)

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)

    # A false negative: actual default (1), predicted no-default (0)
    fn_mask = (y_test == 1) & (y_pred_t == 0)
    # A false positive: actual no-default (0), predicted default (1)
    fp_mask = (y_test == 0) & (y_pred_t == 1)

    total_fn_cost = fn_cost_per_loan[fn_mask].sum()
    total_fp_cost = fp_cost_per_loan[fp_mask].sum()
    total_cost = total_fn_cost + total_fp_cost

    results.append({
        'threshold': round(t, 2),
        'fn_count': fn_mask.sum(),
        'fp_count': fp_mask.sum(),
        'fn_cost': total_fn_cost,
        'fp_cost': total_fp_cost,
        'total_cost': total_cost
    })

cost_df = pd.DataFrame(results)
print(cost_df.to_string(index=False))

best_row = cost_df.loc[cost_df['total_cost'].idxmin()]
print(f"\nOptimal threshold: {best_row['threshold']} (total cost: ${best_row['total_cost']:,.0f})")

# Compare against the default 0.5 cutoff for reference
default_row = cost_df[cost_df['threshold'] == 0.5]
print(f"Cost at default 0.5 threshold: ${default_row['total_cost'].values[0]:,.0f}")