import pandas as pd
from Paths import CLEANED_DIR
from sklearn.model_selection import train_test_split

# Load the feature-engineered dataset and the cleaned dataset.
# The feature-engineered file no longer contains addr_state, so we temporarily recover it from the cleaned dataset to fix target leakage.
df = pd.read_csv(CLEANED_DIR / "loan_features.csv")
cleaned = pd.read_csv(CLEANED_DIR / "loan_cleaned.csv")

# Verify that both datasets contain the same number of rows before rejoining the state information.
print(df.shape[0], cleaned.shape[0])

# loan_features.py only transformed/dropped columns and did not remove rows, so the rows still correspond positionally.
# Reattach addr_state using the original row order.
df['addr_state'] = cleaned['addr_state'].values

print(df[['addr_state', 'addr_state_risk']].head())

# Separate the target variable from the predictor variables.
y = df['default']
X = df.drop(columns=['default'])

# Split the data before recalculating any target-based features.
# Stratification keeps the ~20% default rate approximately consistent between training and test sets, which is important for this imbalanced target.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)
print(y_train.mean(), y_test.mean())  # sanity check: both should be ~0.2007

# --- Fix addr_state_risk leakage ---
# The original addr_state_risk feature was calculated using the complete
# dataset, meaning information from the test set was used to calculate a feature that the model later sees.
# Recalculate the state default rates using TRAINING DATA ONLY.
# This ensures the test set remains completely unseen during model training.
train_state_risk = y_train.groupby(X_train['addr_state']).mean()
overall_train_rate = y_train.mean()

# Apply the training-derived state risk rates to both datasets.
# The test set receives risk information learned only from the training set.
X_train['addr_state_risk'] = X_train['addr_state'].map(train_state_risk)

# If a state appears in the test set but not in training, use the overall
# training default rate as a fallback rather than using test-set information.
X_test['addr_state_risk'] = X_test['addr_state'].map(train_state_risk).fillna(overall_train_rate)

# addr_state was only needed to calculate the leakage-free risk feature.
# Removed it before passing the data to the model.
X_train = X_train.drop(columns=['addr_state'])
X_test = X_test.drop(columns=['addr_state'])

print(X_train['addr_state_risk'].describe())
print(X_test['addr_state_risk'].describe())

# --- Step 4: Save the split sets so modeling can just load them ---
X_train.to_csv(CLEANED_DIR / "X_train.csv", index=False)
X_test.to_csv(CLEANED_DIR / "X_test.csv", index=False)
y_train.to_csv(CLEANED_DIR / "y_train.csv", index=False)
y_test.to_csv(CLEANED_DIR / "y_test.csv", index=False)
print("Saved train/test splits.")
