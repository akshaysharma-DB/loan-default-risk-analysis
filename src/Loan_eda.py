import pandas as pd

df = pd.read_csv(r'D:\Downloads\Data\loan_features.csv')
print(df.shape)
print(df['default'].mean())

#Default rate by loan grade 
# I want to check if the default rate is higher for lower grades as expected. If it is not, then that would be a red flag worth investigating.,
grade_default  = df.groupby('grade')['default'].agg(['mean', 'count'])
print(grade_default)
# an A-grade borrower default about 8X less often than a G-grade borrower which is 6% to 50%. This is as expected and makes sense.

df['dti_bucket'] = pd.cut(df['dti'], bins=[0, 10, 20, 30, 40, 100], labels=['0-10', '10-20', '20-30', '30-40', '40+'])
dti_default = df.groupby('dti_bucket', observed=True)['default'].agg(['mean', 'count'])
print(dti_default)
#Default rate by Financial health indicators (DTI and Income)
df['income_bucket'] = pd.qcut(df["annual_inc"], q=5, labels=['lowest 20%', 'Low-mid', 'Mid', 'Mid-high', 'Highest 20%'])
income_default = df.groupby('income_bucket', observed=True)['default'].agg(['mean', 'count'])
print(income_default)

# Check default rate by loan purpose. We already know grade, DTI, and income
# predict default -this checks whether WHY someone borrowed also matters.
# Purpose columns are one-hot encoded (purpose_medical, purpose_car, etc.),
# so we grab all of them automatically instead of typing each name by hand.
purpose_cols = [col for col in df.columns if col.startswith('purpose_')]
 # Keep only the rows where this specific purpose applies (value == 1)
for col in purpose_cols:
    subset = df[df[col] == 1]
    print(f"{col}: {subset['default'].mean():.3f} (n={len(subset)})")

 # Sanity check for addr_state_risk. We BUILT this feature ourselves earlier
# from the default column, so this isn't new EDA — it's confirming the
# feature actually behaves as expected before we trust it in the model.
# Riskiest states should clearly show a higher default rate than safest.
df['state_risk_bucket'] = pd.qcut(df['addr_state_risk'], q=5,
                                    labels=['Safest 20%', 'Low-mid', 'Mid', 'Mid-high', 'Riskiest 20%'])

state_check = df.groupby('state_risk_bucket', observed=True)['default'].agg(['mean', 'count'])
print(state_check)

