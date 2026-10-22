
# ============================================================
# RAINFALL PREDICTION USING LOGISTIC REGRESSION
# Complete Machine Learning Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

CSV_FILE = "train.csv"
TARGET_COLUMN = "rainfall"

# For the Australian weather dataset, use:
# CSV_FILE = "weatherAUS.csv"
# TARGET_COLUMN = "RainTomorrow"

df = pd.read_csv(CSV_FILE)

print("=" * 55)
print("RAINFALL PREDICTION USING LOGISTIC REGRESSION")
print("=" * 55)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nAvailable columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 2. CLEAN DATASET
# ------------------------------------------------------------

df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' not found.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Change TARGET_COLUMN to the correct column name."
    )

# Remove rows with missing target values
df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Clean text target values
if df[TARGET_COLUMN].dtype == "object":
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype(str).str.strip().str.lower()
    )

# Remove empty target labels
df = df[
    df[TARGET_COLUMN].astype(str).str.strip().ne("")
].copy()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTarget class counts:")
print(df[TARGET_COLUMN].value_counts())

# ------------------------------------------------------------
# 3. PREPARE BINARY TARGET
# ------------------------------------------------------------

classes = list(pd.unique(df[TARGET_COLUMN]))

print("\nTarget classes found:", classes)

if len(classes) != 2:
    raise ValueError(
        f"Expected exactly 2 target classes, but found "
        f"{len(classes)}: {classes}\n"
        "Check TARGET_COLUMN and select the correct binary "
        "rainfall target. Do not use a weather feature as target."
    )

# Map target classes to 0 and 1
# Class 0 = first class shown below
# Class 1 = second class shown below
class_mapping = {classes[0]: 0, classes[1]: 1}

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN].map(class_mapping).astype(int)
)

print("Target mapping:", class_mapping)

# ------------------------------------------------------------
# 4. VISUALIZE TARGET DISTRIBUTION
# ------------------------------------------------------------

df[TARGET_COLUMN].value_counts().sort_index().plot(
    kind="bar", figsize=(6, 4)
)

plt.title("Rainfall Class Distribution")
plt.xlabel("Target class (0 or 1)")
plt.ylabel("Number of records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].copy()

# Remove identifiers, date columns, and columns that may
# directly reveal the answer. Adjust this list for your data.
columns_to_drop = [
    "id", "ID", "Id",
    "Date", "date",
    "RISK_MM", "risk_mm"
]

X = X.drop(
    columns=[col for col in columns_to_drop if col in X.columns],
    errors="ignore"
)

# Remove entirely empty and constant columns
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=False) > 1]

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")

print("\nNumber of features:", X.shape[1])
print("Feature columns:", X.columns.tolist())

# ------------------------------------------------------------
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumerical features:", numerical_columns)
print("Categorical features:", categorical_columns)

# ------------------------------------------------------------
# 7. SPLIT INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

if y.value_counts().min() < 2:
    raise ValueError(
        "Each target class needs at least 2 records "
        "to perform a stratified train/test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 8. PREPROCESS DATA
# ------------------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# ------------------------------------------------------------
# 9. CREATE AND TRAIN LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# ------------------------------------------------------------
# 10. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 actual values:")
print(y_test.iloc[:10].tolist())

print("\nFirst 10 predicted values:")
print(y_pred[:10].tolist())

# ------------------------------------------------------------
# 11. EVALUATE MODEL ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            f"Class 0 ({classes[0]})",
            f"Class 1 ({classes[1]})"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

confusion_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        str(classes[0]),
        str(classes[1])
    ]
)

confusion_display.plot()
plt.title("Rainfall Prediction - Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 13. ROC-AUC SCORE
# ------------------------------------------------------------

if y_test.nunique() == 2:
    auc = roc_auc_score(y_test, y_probability)
    print(f"ROC-AUC Score: {auc:.4f}")
else:
    print("ROC-AUC could not be calculated.")

# ------------------------------------------------------------
# 14. PREDICT USING ONE TEST SAMPLE
# ------------------------------------------------------------

sample = X_test.iloc[[0]]

sample_prediction = model.predict(sample)[0]
sample_probabilities = model.predict_proba(sample)[0]

print("\n" + "=" * 55)
print("SAMPLE RAINFALL PREDICTION")
print("=" * 55)

print("\nWeather information:")
print(sample.to_string(index=False))

print("\nPredicted class:", sample_prediction)
print("Predicted result:", classes[sample_prediction])

print(
    f"Probability of {classes[0]}: "
    f"{sample_probabilities[0] * 100:.2f}%"
)

print(
    f"Probability of {classes[1]}: "
    f"{sample_probabilities[1] * 100:.2f}%"
)

# ------------------------------------------------------------
# 15. SAVE TRAINED MODEL
# ------------------------------------------------------------

joblib.dump(
    {
        "model": model,
        "class_mapping": class_mapping,
        "target_column": TARGET_COLUMN,
        "feature_columns": X.columns.tolist()
    },
    "rainfall_logistic_regression.pkl"
)

print("\nTrained model saved as:")
print("rainfall_logistic_regression.pkl")

# ------------------------------------------------------------
# 16. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("PROJECT SUMMARY")
print("=" * 55)

print("Project: Rainfall Prediction")
print("Algorithm: Logistic Regression")
print("Target column:", TARGET_COLUMN)
print("Classes:", classes)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Status: Completed")
