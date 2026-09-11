import pandas as pd
from Paths import RAW_DIR, CLEANED_DIR

# Load the raw Lending Club dataset.
# This is the starting point of the pipeline before any filtering or cleaning.
df = pd.read_csv(RAW_DIR / "loan.csv", low_memory=False)

print(df.shape)
print(df.head())
print(df.info())
print(df.tail())
print(df['loan_status'].value_counts(dropna=False))

# Keep only loans with a known final outcome.
# Ongoing loans are excluded because their default status is not known yet.
df_model = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
print(df_model.shape)
print(df_model['loan_status'].value_counts())
print(df_model.columns.tolist())

# Columns that only exist after a loan is issued eg. payment history, settlememts.
# A lender would never have this info before lending. So, using these wouldn't make sense to build this model.
leakage_cols = [
    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'acc_now_delinq', 'hardship_flag', 'hardship_type', 'hardship_reason',
    'hardship_status', 'deferral_term', 'hardship_amount',
    'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date',
    'hardship_length', 'hardship_dpd', 'hardship_loan_status',
    'orig_projected_additional_accrued_interest',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    'debt_settlement_flag', 'debt_settlement_flag_date',
    'settlement_status', 'settlement_date', 'settlement_amount',
    'settlement_percentage', 'settlement_term'
]

# Remove identifiers and descriptive fields that are not useful as predictive features
# and could introduce noise into the model. These are not leakage, but they don't help the model.
noise_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code', 'policy_code']

df_model = df_model.drop(columns=leakage_cols + noise_cols)
print(df_model.shape)

# Measure missingness in the remaining features before deciding
# which columns should be removed or attributed filled. This is a key step in data cleaning and preparation for modeling.
missing = df_model.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_model)) * 100
print(missing_pct[missing_pct > 0])
pd.set_option('display.max_rows', None)
print(missing_pct[missing_pct > 0])

# Remove features with more than 40% missing values because they are
# too sparse to provide reliable information without aggressive imputation.
# Features below this threshold are retained and handled separately.
high_missing_cols = missing_pct[missing_pct > 40].index.tolist()
print(f"Dropping {len(high_missing_cols)} columns with >40% missing:")
print(high_missing_cols)

df_model = df_model.drop(columns=high_missing_cols)
print(df_model.shape)

# Fill missing numeric values with the median.
# Median imputation is less sensitive to extreme values than the mean,
# which is useful for skewed financial variables.
numeric_cols = df_model.select_dtypes(include='number').columns
df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

print(df_model.isnull().sum().sum())

# Employment length is categorical text, so numeric imputation does not apply.
# Keep missing values explicitly as "Unknown" rather than assuming an employment duration that was never provided.
df_model['emp_length'] = df_model['emp_length'].fillna('Unknown')

print(df_model['emp_length'].value_counts())
print(df_model.isnull().sum().sum())

# Save the cleaned dataset as the input for the feature-engineering stage.
df_model.to_csv(r'D:\Downloads\Data\loan_cleaned.csv', index=False)
print("Saved cleaned data.")
from Paths import RAW_DIR, CLEANED_DIR

# Just verifying the paths exist and are correct before uploading to GitHub.
print("Raw:", RAW_DIR)
print("Cleaned:", CLEANED_DIR)
print("Raw exists:", RAW_DIR.exists())
print("Cleaned exists:", CLEANED_DIR.exists())
