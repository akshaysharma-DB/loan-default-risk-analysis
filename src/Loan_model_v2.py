import pandas as pd
from Paths import CLEANED_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Load the train/test data created earlier so the model uses the same
# fixed split for training and evaluation.
X_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
X_test = pd.read_csv(CLEANED_DIR / "X_test.csv")
y_train = pd.read_csv(CLEANED_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(CLEANED_DIR / "y_test.csv").squeeze()

print(X_train.shape, X_test.shape)


# Scale the features because they are measured on very different ranges.
# Putting them on a similar scale helps logistic regression optimize
# the model more reliably.
scaler = StandardScaler()

# Fit the scaler only on training data so information from the test set
# does not influence the model during training.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Defaults are the minority class, so class weighting gives more
# importance to correctly identifying default cases.
# This addresses the problem seen in the baseline, where high accuracy
# came mainly from predicting non-default loans.
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)


# Generate predictions on the unseen test data to see how the improved
# model performs on loans it has not seen before.
y_pred = model.predict(X_test_scaled)


# Accuracy is reported for reference, but the confusion matrix and
# classification report are more useful because identifying actual
# defaults is the main concern in this analysis.
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Keep the original feature names so the model coefficients can be
# linked back to the variables used in the risk analysis.
coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': model.coef_[0]
})


# Sort by absolute coefficient size to identify the features with
# the strongest relationship with the model's predicted default risk.
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)

print(coefficients.head(15).to_string(index=False))


from sklearn.metrics import roc_auc_score


# Use predicted probabilities instead of only yes/no predictions.
# This measures how well the model ranks loans by risk across
# different possible probability thresholds.
y_proba = model.predict_proba(X_test_scaled)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {auc:.4f}")


import numpy as np


# A 0.5 threshold is not automatically the best choice for lending
# because false negatives and false positives can have different
# financial consequences. Several thresholds are compared below.
fn_cost_per_loan = X_test['loan_amnt']


# A false positive represents a potentially profitable loan that is
# rejected. The estimated cost uses one year of expected interest
# as a simple business assumption.
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


# Select the threshold with the lowest estimated total cost instead
# of automatically using the standard 0.5 classification cutoff.
best_row = cost_df.loc[cost_df['total_cost'].idxmin()]
print(f"\nOptimal threshold: {best_row['threshold']} (total cost: ${best_row['total_cost']:,.0f})")


# Compare the optimized threshold with 0.5 to show whether changing
# the decision threshold improves the estimated business outcome.
default_row = cost_df[cost_df['threshold'] == 0.5]
print(f"Cost at default 0.5 threshold: ${default_row['total_cost'].values[0]:,.0f}")
