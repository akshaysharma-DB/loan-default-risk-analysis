import pandas as pd
from Paths import CLEANED_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the train/test data created in the previous step.
# Keeping the split separate makes the model evaluation consistent and reproducible.
X_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
X_test = pd.read_csv(CLEANED_DIR / "X_test.csv")
y_train = pd.read_csv(CLEANED_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(CLEANED_DIR / "y_test.csv").squeeze()

print(X_train.shape, X_test.shape)

# Start with a simple Logistic Regression model to establish a baseline.
# Scaling and class weighting are intentionally left out so the improvements in the next model can be compared against this starting point.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Generate predictions on the unseen test data to evaluate how well the baseline model performs on loans it has not seen during training.
y_pred = model.predict(X_test)

# Accuracy alone can be misleading here because defaults are the minority class.
# The confusion matrix and classification report show how well the model actually identifies default cases.
print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))
