"""
train_model.py
---------------
1. Loads student_placement_career_success_dataset_2026.csv
2. Cleans / preprocesses data (missing values, encoding, scaling)
3. Performs EDA and saves plots to the eda_plots/ folder
4. Trains & compares Logistic Regression, Decision Tree, Random Forest
5. Saves the best model + preprocessing objects to disk (model.pkl etc.)
   so that app.py can load them for live predictions.

NOTE ON THE TARGET COLUMN
--------------------------
This dataset's original `Placement_Status` column is constant (every row
is "placed"), so it carries no signal and cannot be used as a classification
target. Instead, we derive a meaningful target from `Salary_LPA`: students
at or above the median salary are labeled "High Package", the rest
"Standard Package". This keeps the same placement-prediction framing while
actually being learnable from the data. `Salary_LPA` itself is then
excluded from the model's input features, since it's the source of the
label and using it as a feature would leak the answer.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# =================================================================
# COLUMN CONFIG  -> edit these if you plug in a different CSV
# =================================================================
DATA_PATH = "student_placement_career_success_dataset_2026.csv"
TARGET_COL = "Placement_Package"          # derived: "High Package" / "Standard Package"
SALARY_COL = "Salary_LPA"                 # used only to build the target, then dropped from features
UNUSED_COLS = ["Placement_Status"]        # constant in this dataset, carries no information

NUMERIC_COLS = [
    "Age", "CGPA", "DSA_Problems_Solved", "Internships", "Certifications",
    "Projects_Count", "Communication_Skills", "Aptitude_Test_Score",
    "LeetCode_Rating", "GitHub_Contributions", "Hackathons_Participated",
    "AI_ML_Skill_Level", "System_Design_Knowledge", "Resume_Score",
    "Mock_Interview_Score",
]
CATEGORICAL_COLS = ["Gender", "College_Tier", "Specialization"]

os.makedirs("eda_plots", exist_ok=True)

# =================================================================
# 1. LOAD DATA
# =================================================================
df = pd.read_csv(DATA_PATH)
print("Loaded data:", df.shape)
print(df.head())

df = df.drop(columns=[c for c in UNUSED_COLS if c in df.columns])

# =================================================================
# 2. HANDLE MISSING VALUES
# =================================================================
print("\nMissing values before cleaning:\n", df.isnull().sum())

for col in NUMERIC_COLS:
    df[col] = df[col].fillna(df[col].median())

for col in CATEGORICAL_COLS:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:\n", df.isnull().sum().sum(), "total nulls")

# =================================================================
# 3. BUILD THE CLASSIFICATION TARGET FROM SALARY_LPA
# =================================================================
salary_median = df[SALARY_COL].median()
df[TARGET_COL] = np.where(
    df[SALARY_COL] >= salary_median, "High Package", "Standard Package"
)
print(f"\nSalary median used as cutoff: {salary_median:.2f} LPA")
print(df[TARGET_COL].value_counts())

# =================================================================
# 4. EXPLORATORY DATA ANALYSIS (plots saved to eda_plots/)
# =================================================================
sns.set(style="whitegrid")

# Target balance
plt.figure(figsize=(5, 4))
sns.countplot(x=TARGET_COL, data=df, palette="viridis")
plt.title("Placement Package Distribution")
plt.tight_layout()
plt.savefig("eda_plots/target_distribution.png")
plt.close()

# CGPA vs package
plt.figure(figsize=(6, 4))
sns.boxplot(x=TARGET_COL, y="CGPA", data=df, palette="viridis")
plt.title("CGPA vs Placement Package")
plt.tight_layout()
plt.savefig("eda_plots/cgpa_vs_placement.png")
plt.close()

# Correlation heatmap (numeric features only, Salary_LPA excluded on purpose)
plt.figure(figsize=(10, 8))
corr = df[NUMERIC_COLS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

# Internships vs high-package rate
plt.figure(figsize=(6, 4))
sns.barplot(
    x="Internships", y=(df[TARGET_COL] == "High Package").astype(int),
    data=df, palette="viridis"
)
plt.ylabel("High Package Rate")
plt.title("High Package Rate by Number of Internships")
plt.tight_layout()
plt.savefig("eda_plots/internships_vs_placement.png")
plt.close()

# AI/ML skill vs package (the strongest correlate with salary in this dataset)
plt.figure(figsize=(6, 4))
sns.boxplot(x=TARGET_COL, y="AI_ML_Skill_Level", data=df, palette="viridis")
plt.title("AI/ML Skill Level vs Placement Package")
plt.tight_layout()
plt.savefig("eda_plots/aiml_skill_vs_placement.png")
plt.close()

print("\nEDA plots saved in eda_plots/ folder.")

# =================================================================
# 5. ENCODING CATEGORICAL FEATURES
# =================================================================
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # remember mapping for the Streamlit app

target_le = LabelEncoder()
df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])
print("\nTarget classes:", list(target_le.classes_), "->", list(target_le.transform(target_le.classes_)))

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # Salary_LPA intentionally excluded
X = df[FEATURE_COLS]
y = df[TARGET_COL]

# =================================================================
# 6. TRAIN / TEST SPLIT + SCALING
# =================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =================================================================
# 7. TRAIN & COMPARE MODELS
# =================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    ),
}

results = []
best_model_name = None
best_model = None
best_f1 = -1

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1,
    })

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=target_le.classes_))

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\nModel comparison:\n", results_df)
results_df.to_csv("model_comparison.csv", index=False)

# Confusion matrix for the best model
best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("eda_plots/confusion_matrix_best_model.png")
plt.close()

print(f"\nBest model: {best_model_name}  (F1 = {best_f1:.3f})")

# =================================================================
# 8. SAVE MODEL + PREPROCESSING OBJECTS FOR THE STREAMLIT APP
# =================================================================
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_le, "target_encoder.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
joblib.dump(best_model_name, "best_model_name.pkl")
joblib.dump(salary_median, "salary_median.pkl")

print("\nSaved: model.pkl, scaler.pkl, label_encoders.pkl, "
      "target_encoder.pkl, feature_cols.pkl, best_model_name.pkl, salary_median.pkl")
print("You can now run: streamlit run app.py")