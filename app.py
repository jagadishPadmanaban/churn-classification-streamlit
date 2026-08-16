"""
Streamlit demo app for the Telco Customer Churn classification project.

Lets a user upload a test CSV (features + true 'Churn' label), pick one
of the 5 trained models, and see evaluation metrics + a confusion
matrix / classification report computed live on the uploaded data.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_DATA = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

TARGET_COL = "Churn"

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")


@st.cache_resource
def load_model(filename: str):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_training_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Telco Customer Churn — Classification Demo")
st.write(
    "Upload a test CSV (same schema as `test_data.csv`, including the true "
    "`Churn` column), pick a model, and see how it performs."
)

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    use_sample = st.checkbox("Use bundled sample test_data.csv", value=uploaded_file is None)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif use_sample:
    data = pd.read_csv(DEFAULT_TEST_DATA)
else:
    st.info("Upload a CSV or check 'Use bundled sample test_data.csv' to continue.")
    st.stop()

st.subheader("Preview of test data")
st.dataframe(data.head(10), width="stretch")

has_labels = TARGET_COL in data.columns
if not has_labels:
    st.warning(
        f"No '{TARGET_COL}' column found — predictions will be shown, "
        "but evaluation metrics require the true label column."
    )

pipeline = load_model(MODEL_FILES[model_name])

X = data.drop(columns=[TARGET_COL]) if has_labels else data
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)[:, 1]

result_df = data.copy()
result_df["Predicted"] = np.where(predictions == 1, "Yes", "No")
result_df["Churn Probability"] = probabilities.round(4)

st.subheader(f"Predictions — {model_name}")
st.dataframe(result_df.head(20), width="stretch")

if has_labels:
    y_true = data[TARGET_COL].map({"Yes": 1, "No": 0})
    metrics = evaluate(y_true, predictions, probabilities)

    st.subheader("Evaluation Metrics on Uploaded Data")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_b:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, predictions, target_names=["No", "Yes"])
        st.text(report)

st.divider()
st.subheader("Training-time Comparison Across All 5 Models")
st.caption("Metrics computed on the held-out 20% split during model training.")
training_metrics = load_training_metrics()
comparison_df = pd.DataFrame(training_metrics).T
comparison_df.index.name = "ML Model Name"
st.dataframe(comparison_df.style.format("{:.4f}"), width="stretch")
