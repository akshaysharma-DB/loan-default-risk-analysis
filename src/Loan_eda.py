import pandas as pd
from Paths import CLEANED_DIR

# Load the feature-engineered dataset created in the previous pipeline stage.
df = pd.read_csv(CLEANED_DIR / "loan_features.csv")
print(df.shape)
print(df['default'].mean())

# Check whether loan grade is strongly associated with default risk.
# Grade has an ordered risk scale, so we expect default rates to increase
# as the grade moves from A (lowest risk) toward G (highest risk).
# A result that contradicts this pattern would be a useful data-quality indicator.
grade_default  = df.groupby('grade')['default'].agg(['mean', 'count'])
print(grade_default)

# Group borrowers by Debt-to-Income ratio (DTI) to examine whether
# higher debt burden is associated with higher default risk.
# Bucketing makes the relationship easier to compare across risk levels.
df['dti_bucket'] = pd.cut(df['dti'], bins=[0, 10, 20, 30, 40, 100], labels=['0-10', '10-20', '20-30', '30-40', '40+'])
dti_default = df.groupby('dti_bucket', observed=True)['default'].agg(['mean', 'count'])
print(dti_default)

# Compare default rates across borrower income groups.
# Quantile-based buckets create groups with roughly similar numbers of borrowers, making the comparison less sensitive to the skew
# typically present in income data.
df['income_bucket'] = pd.qcut(df["annual_inc"], q=5, labels=['lowest 20%', 'Low-mid', 'Mid', 'Mid-high', 'Highest 20%'])
income_default = df.groupby('income_bucket', observed=True)['default'].agg(['mean', 'count'])
print(income_default)

# Check whether the reason for borrowing is associated with different default rates. Grade, DTI, and income capture borrower risk, while
# loan purpose may provide additional information about borrowing behavior.
# The purpose variables are one-hot encoded, so identify them automatically instead of listing every category manually.
purpose_cols = [col for col in df.columns if col.startswith('purpose_')]

# Only include borrowers for whom this particular loan purpose applies.
for col in purpose_cols:
    subset = df[df[col] == 1]
    print(f"{col}: {subset['default'].mean():.3f} (n={len(subset)})")

# Validate the state-risk feature created during feature engineering.
# States were converted into historical default-risk values, so we check
# whether borrowers in higher-risk buckets actually have higher default rates.
# This is a sanity check before relying on the feature in the model.
df['state_risk_bucket'] = pd.qcut(df['addr_state_risk'], q=5,
                                  labels=['Safest 20%', 'Low-mid', 'Mid', 'Mid-high', 'Riskiest 20%'])

state_check = df.groupby('state_risk_bucket', observed=True)['default'].agg(['mean', 'count'])
print(state_check)
