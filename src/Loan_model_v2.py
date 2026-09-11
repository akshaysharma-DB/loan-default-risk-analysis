import pandas as pd
from Paths import CLEANED_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the train/test data created earlier so the model is trained
# and evaluated on the same fixed split.
X_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
X_test = pd.read_csv(CLEANED_DIR / "X_test.csv")
y_train = pd.read_csv(CLEANED_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(CLEANED_DIR / "y_test.csv").squeeze()

print(X_train.shape, X_test.shape)

# --- Feature scaling ---
# Logistic regression's optimizer struggles when features are on very
# different scales (loan_amnt in thousands vs dti in single digits vs
# 0/1 dummy columns). StandardScaler transforms each column to mean=0,
# std=1, which fixes the convergence warning and helps the model treat
# each feature fairly during optimization.
#
# IMPORTANT: fit the scaler on X_train ONLY, then use that same fitted
# scaler to transform X_test. If we fit on the full dataset (or on test too), information about the test set's distribution leaks into
# training -- the same leakage principle as addr_state_risk earlier.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defaults are the minority class, so class weighting gives more
# importance to correctly identifying default cases.
# This helps address the problem seen in the baseline model, where
# high accuracy came mainly from predicting non-default loans.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Generate predictions on the unseen test data to see how theimproved model performs on loans it has not seen before.
y_pred = model.predict(X_test_scaled)

# Accuracy is reported for reference, but the confusion matrix and
# classification report are more useful here because identifying
# actual defaults is the main business concern.
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

import pandas as pd

# Keep the original feature names so the model coefficients can be
# linked back to the variables used in the lending-risk analysis.
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
})

# Sort by absolute coefficient size to see which features have the
# strongest relationship with the model's predicted default risk.
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)

print(coefficients.head(15).to_string(index=False))

from sklearn.metrics import roc_auc_score

# Use predicted probabilities instead of only yes/no predictions.
# This allows the model's ability to rank borrowers by risk to be
# measured independently of a single classification threshold.
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 (default)

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")


import numpy as np

# The default 0.5 threshold is not necessarily the best choice for
# lending decisions because false negatives and false positives
# can have very different financial consequences.
# Test several thresholds and compare their estimated costs.
fn_cost_per_loan = X_test['loan_amnt']

# A false positive means rejecting a loan that would not have defaulted.
# The estimated cost is the interest that could have been earned
# from approving that loan, using one year of interest as a simple assumption.
fp_cost_per_loan = X_test['loan_amnt'] * (X_test['int_rate'] / 100)

results = []
thresholds = np.arange(0.1, 0.95, 0.05)

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)

    # Actual defaults predicted as non-defaults are false negatives.
    # These represent loans where the model failed to flag the risk.
    fn_mask = (y_test == 1) & (y_pred_t == 0)

    # Actual non-defaults predicted as defaults are false positives.
    # These represent potentially profitable loans that would be rejected.
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

# Select the threshold with the lowest estimated total cost.
# This gives a business-oriented decision point instead of simply
# relying on the standard 0.5 classification cutoff.
best_row = cost_df.loc[cost_df['total_cost'].idxmin()]
print(f"\nOptimal threshold: {best_row['threshold']} (total cost: ${best_row['total_cost']:,.0f})")

# Compare the optimized threshold with the standard 0.5 cutoff
# to show whether changing the decision threshold actually improves
# the estimated business outcome.
default_row = cost_df[cost_df['threshold'] == 0.5]
print(f"Cost at default 0.5 threshold: ${default_row['total_cost'].values[0]:,.0f}")
