import pandas as pd
from sklearn.model_selection import train_test_split

# --- Step 1: Load and rejoin addr_state ---
df = pd.read_csv(r'D:\Downloads\Data\loan_features.csv')
cleaned = pd.read_csv(r'D:\Downloads\Data\loan_cleaned.csv')

# Sanity check: same row count (should both be 1303607)
print(df.shape[0], cleaned.shape[0])

# Rows line up positionally since loan_features.py never filtered rows,
# only transformed/dropped columns -- so we can rejoin by position.
df['addr_state'] = cleaned['addr_state'].values

print(df[['addr_state', 'addr_state_risk']].head())

# --- Step 2: Train/test split ---
y = df['default']
X = df.drop(columns=['default'])

# stratify=y keeps the ~20% default rate consistent in both sets,
# since it's not a 50/50 target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)
print(y_train.mean(), y_test.mean())  # sanity check: both should be ~0.2007

# --- Step 3: Fix addr_state_risk leakage ---
# Recalculate using ONLY training data -- this is the actual leakage fix.
# The old addr_state_risk column (built on the full dataset) gets replaced
# with this one before modeling.
train_state_risk = y_train.groupby(X_train['addr_state']).mean()
overall_train_rate = y_train.mean()

X_train['addr_state_risk'] = X_train['addr_state'].map(train_state_risk)
# .fillna handles any state that appears in test but not in train (rare,
# but possible for small states) -- falls back to the overall train rate
X_test['addr_state_risk'] = X_test['addr_state'].map(train_state_risk).fillna(overall_train_rate)

# Drop addr_state again since we only needed it to recompute the risk score
X_train = X_train.drop(columns=['addr_state'])
X_test = X_test.drop(columns=['addr_state'])

print(X_train['addr_state_risk'].describe())
print(X_test['addr_state_risk'].describe())

# --- Step 4: Save the split sets so modeling can just load them ---
X_train.to_csv(r'D:\Downloads\Data\X_train.csv', index=False)
X_test.to_csv(r'D:\Downloads\Data\X_test.csv', index=False)
y_train.to_csv(r'D:\Downloads\Data\y_train.csv', index=False)
y_test.to_csv(r'D:\Downloads\Data\y_test.csv', index=False)
print("Saved train/test splits.")