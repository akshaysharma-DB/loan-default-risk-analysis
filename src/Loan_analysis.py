import pandas as pd

# Load the raw dataset (2,26 million loans, 145 columns)
df = pd.read_csv(r'D:\Downloads\Data\loan.csv', low_memory=False)

print(df.shape)
print(df.head())
print(df.info())
print(df.tail())
print(df['loan_status'].value_counts(dropna=False))

# Keep only loans with a finished outcome
#  as we can't keep those which are on progress yet because we don't know if they will default.
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

# Columns that are just identifiers not leakage eg.  Id, policy  code etc. are just not useful for prediction.
noise_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title',
              'zip_code', 'policy_code']

df_model = df_model.drop(columns=leakage_cols + noise_cols)
print(df_model.shape)

# Check how much data is missing in each remaining column
missing = df_model.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_model)) * 100
print(missing_pct[missing_pct > 0])
pd.set_option('display.max_rows', None)
print(missing_pct[missing_pct > 0])

#  Drop columns that have 40% values missing of  rows because they are too sparse for this model.
# Columns below this threshold get their gaps filled insted of dropped.
high_missing_cols = missing_pct[missing_pct > 40].index.tolist()
print(f"Dropping {len(high_missing_cols)} columns with >40% missing:")
print(high_missing_cols)

df_model = df_model.drop(columns=high_missing_cols)
print(df_model.shape)

# Fill remaining gaps in numeric columns with median
# Because median is less skewed by extreme values. 
numeric_cols = df_model.select_dtypes(include='number').columns
df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

print(df_model.isnull().sum().sum())
# emp_length is text, not numeric, so it needs its own fill
#label missing values explicitly as Unknown rather than guessing.
df_model['emp_length'] = df_model['emp_length'].fillna('Unknown')

print(df_model['emp_length'].value_counts())
print(df_model.isnull().sum().sum())

# Saved this 
df_model.to_csv(r'D:\Downloads\Data\loan_cleaned.csv', index=False)
print("Saved cleaned data.")
