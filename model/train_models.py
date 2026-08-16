"""
Trains 5 classification models on the Telco Customer Churn dataset,
evaluates them, and saves the fitted pipelines + metrics for the
Streamlit app to consume.

Run from the project root:
    python model/train_models.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_COL = "Churn"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save the held-out test split (with the true label) for the repo
    # submission and for use inside the Streamlit app.
    test_export = X_test.copy()
    test_export[TARGET_COL] = y_test.map({1: "Yes", 0: "No"}).values
    test_export.to_csv(TEST_DATA_PATH, index=False)

    preprocessor = build_preprocessor(X_train)
    models = get_models()

    metrics = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics[name] = evaluate(y_test, y_pred, y_proba)

        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{filename}.pkl"))
        print(f"[saved] {filename}.pkl -> {metrics[name]}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nComparison table:")
    print(pd.DataFrame(metrics).T)


if __name__ == "__main__":
    main()
