import pandas as pd

# Load the alrady cleaned dataset
df = pd.read_csv(r'D:\Downloads\Data\loan_cleaned.csv')

print(df.shape)

# Use variable to make model work for this loan status text column
# 1= Defaulted, 0 = fully paid.
df['default'] = (df['loan_status'] == 'Charged Off').astype(int)
print(df['default'].value_counts())
print(df['default'].mean())

#Extract only numbers from text like "36 months" will be used as 36.
df['term'] = df['term'].str.extract(r'(\d+)').astype(int)
df['initial_list_status'] = (df['initial_list_status'] == 'w').astype(int)
df['application_type'] = (df['application_type'] == 'Joint App').astype(int)
df['disbursement_method'] = (df['disbursement_method'] == 'DirectPay').astype(int)

# Loan grade has order from A(best) to G(worst)
# so we use it numbers instead as 1(best) to 7 (worst)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade'] = df['grade'].map(grade_map)

# same idea is for sub_grade (A1-G5)
sub_grades = sorted(df['sub_grade'].unique())
sub_grade_map = {sg: i+1 for i, sg in enumerate(sub_grades)}
df['sub_grade'] = df['sub_grade'].map(sub_grade_map)

print(df[['term', 'initial_list_status', 'application_type', 'disbursement_method', 'grade', 'sub_grade']].head(10))

# Emp_length is num+text  so we map it manually.
# but unknown gets -1 so that model can tell "no data" as Emp_length can never be negative
emp_length_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
    '10+ years': 10, 'Unknown': -1
}
df['emp_length'] = df['emp_length'].map(emp_length_map)

print(df['emp_length'].value_counts())

# Categories taht have no order or ranking. So one encode- turn each category into its own 0/1 column
df = pd.get_dummies(df, columns=['home_ownership', 'verification_status', 'purpose'], drop_first=True)

print(df.shape)
print([col for col in df.columns if 'home_ownership' in col or 'verification' in col or 'purpose' in col])

# Convert text dates into real date time objects for python to work upon
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')

# How many years of credit history
df['credit_history_years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365

print(df[['issue_d', 'earliest_cr_line', 'credit_history_years']].head(10))
print(df['credit_history_years'].describe())

#Instead of encoding 50 states, replaced each state with its historical default rate.
#(only used  here on training data for simplicity, to avoid leakage)
state_default_rate = df.groupby('addr_state')['default'].mean()
df['addr_state_risk'] = df['addr_state'].map(state_default_rate)

print(state_default_rate.sort_values(ascending=False).head(10))
print(df[['addr_state', 'addr_state_risk']].head(10))
# drop original text raw/ columns as we already have their numeric replacements
df = df.drop(columns=['addr_state', 'pymnt_plan', 'loan_status', 'issue_d', 'earliest_cr_line'])

print(df.shape)
print(df.dtypes.value_counts())

#saved at last (❁´◡`❁)
df.to_csv(r'D:\Downloads\Data\loan_features.csv', index=False)
print("Saved feature-engineered data.")