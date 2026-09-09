import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Load the split data saved earlier ---
X_train = pd.read_csv(r'D:\Downloads\Data\X_train.csv')
X_test = pd.read_csv(r'D:\Downloads\Data\X_test.csv')
y_train = pd.read_csv(r'D:\Downloads\Data\y_train.csv').squeeze()
y_test = pd.read_csv(r'D:\Downloads\Data\y_test.csv').squeeze()

print(X_train.shape, X_test.shape)

# --- Baseline logistic regression: no scaling, no class weighting ---
# max_iter raised from the default (100) because logistic regression on 83 unscaled features often doesn't converge in time otherwise --
# we're not fixing the scaling issue yet, just letting it finish running
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))