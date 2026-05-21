# ================================================================
# 1. IMPORTS
# ================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

print("Imports ready")

df = pd.read_csv('../women_autonomy_regression/outputs/recoded_analysis_data.csv')

# Drop ONLY truly useless columns
drop_cols = ['weight', 'cluster_id', 'age_group', 'children_group', 'anc_visits']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Remove rare target class
df = df[df['fp_decision_autonomy'] != 6]

print("Data shape:", df.shape)


NUMERICAL_FEATURES = ['age', 'num_children']

TARGET = 'fp_decision_autonomy'

CATEGORICAL_FEATURES = [
    'residence','edu_woman','edu_husband','religion','region',
    'marital_status','wealth','marriage_type',
    'woman_working','fertility_preference','current_method',
    'husband_working','husband_desired_children','anc_group',
    'fieldworker_fp','facility_fp','media_any'
]

CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in df.columns]

# Encode target
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERICAL_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVM": SVC(probability=True, class_weight="balanced")
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted"
    )

    results[name] = {
        "mean": scores.mean(),
        "std": scores.std()
    }

    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

best_model_name = max(results, key=lambda k: results[k]["mean"])
print("\nBest model:", best_model_name)

best_model = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()


# ================================================================
# SAVE EVERYTHING
# ================================================================

import joblib
import json
from datetime import datetime

joblib.dump(final_pipeline, "best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(le, "label_encoder.pkl")

results_dict = {
    "best_model": best_model_name,
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    "timestamp": str(datetime.now())
}

with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=4)

# Save model_results.json (expected by Streamlit app)
model_results = {
    "best_model": best_model_name,
    "cv_performance": {
        "mean_f1": float(results[best_model_name]["mean"]),
        "std_f1": float(results[best_model_name]["std"])
    },
    "test_performance": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted"))
    },
    "confusion_matrix": cm.tolist(),
    "sample_size": len(X),
    "train_size": len(X_train),
    "test_size": len(X_test)
}

with open("model_results.json", "w") as f:
    json.dump(model_results, f, indent=4)

print("\n[SUCCESS] ALL FILES SAVED:")
print(" - best_model.pkl")
print(" - preprocessor.pkl")
print(" - label_encoder.pkl")
print(" - results.json")
print(" - model_results.json")