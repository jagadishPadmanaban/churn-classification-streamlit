# Telco Customer Churn — Classification & Streamlit App

## a. Problem Statement

Customer churn (a customer discontinuing a subscription service) is one of the
costliest problems for subscription-based businesses such as telecom
providers — acquiring a new customer is far more expensive than retaining an
existing one. This project builds and compares five classification models
that predict whether a telecom customer will churn ("Yes"/"No") based on
their demographic details, account information, and the services they have
subscribed to. The best-performing model is then exposed through an
interactive Streamlit web app so predictions and evaluation metrics can be
explored on demand.

## b. Dataset Description

- **Name:** Telco Customer Churn
- **Source:** Public dataset originally released by IBM, mirrored on
  [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  and on GitHub (`IBM/telco-customer-churn-on-icp4d`).
- **Type:** Binary classification (`Churn`: Yes / No)
- **Instances:** 7,043 customers (7,032 retained after dropping rows with
  missing `TotalCharges`) — well above the minimum of 500.
- **Features:** 19 input features (after dropping the `customerID`
  identifier) — well above the minimum of 12. These include:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account info: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
    `MonthlyCharges`, `TotalCharges`
  - Subscribed services: `PhoneService`, `MultipleLines`,
    `InternetService`, `OnlineSecurity`, `OnlineBackup`,
    `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target:** `Churn` — whether the customer left within the last month.
- **Preprocessing:** Rows with a blank `TotalCharges` were dropped (11
  rows). Categorical columns were one-hot encoded and numeric columns were
  standardized inside a single `sklearn` `Pipeline`/`ColumnTransformer`, so
  the exact same transformation is applied at training and inference time.
  Data was split 80/20 (stratified) into train and test sets
  (`random_state=42`).

## c. GitHub Repository Link

https://github.com/jagadishPadmanaban/churn-classification-streamlit

## d. Models Used

All 6 models were trained on the same preprocessed dataset and evaluated on
the same held-out 20% test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8038 | 0.8359 | 0.6485 | 0.5722 | 0.6080 | 0.4795 |
| Decision Tree | 0.7690 | 0.7916 | 0.5679 | 0.5481 | 0.5578 | 0.4017 |
| kNN | 0.7733 | 0.8156 | 0.5741 | 0.5695 | 0.5718 | 0.4176 |
| Naive Bayes | 0.6823 | 0.8049 | 0.4472 | 0.8262 | 0.5803 | 0.4033 |
| Random Forest (Ensemble) | 0.7910 | 0.8360 | 0.6418 | 0.4840 | 0.5518 | 0.4262 |

*(Regenerate this table any time by running `python model/train_models.py`,
which writes the same numbers to `model/metrics.json`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer — highest Accuracy (0.8038), F1 (0.608) and MCC (0.4795). Churn in this dataset correlates strongly and fairly linearly with a handful of features (contract type, tenure, monthly charges), which plays to the strength of a linear model. |
| Decision Tree | Weakest performer on Accuracy, F1 and MCC. A single tree (even depth-limited to 8) tends to overfit the noisy, high-cardinality one-hot encoded categorical splits and doesn't generalize as well as the ensemble/linear alternatives here. |
| kNN | Middling performance, close to the Decision Tree. Distance-based classification is diluted by the large number of one-hot encoded sparse dimensions (curse of dimensionality), so it can't separate classes as cleanly as Logistic Regression. |
| Naive Bayes | Lowest Accuracy and Precision, but by far the highest Recall (0.8262). The conditional-independence assumption is clearly violated (e.g., `InternetService`, `OnlineSecurity`, `TechSupport` are correlated), which biases it toward over-predicting the churn class — useful if the business cost of missing a churner is high, at the cost of many false alarms. |
| Random Forest (Ensemble) | Ties Logistic Regression for the highest AUC (0.836) and has the best Precision, but its Recall is the lowest of all 5 models — it is the most conservative about flagging a customer as "will churn." Good if false positives (unnecessary retention offers) are costly. |
| **Overall Winner for your dataset?** | **Logistic Regression** — best balance of all six metrics (highest Accuracy, F1 and MCC, tied for highest AUC) despite being the simplest model, making it the most practical default for this dataset. |

## Project Structure

```
churn-classification-streamlit/
├── app.py                     # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv              # held-out test split (used by the app + submission)
├── data/
│   └── Telco-Customer-Churn.csv   # full raw dataset used for training
└── model/
    ├── train_models.py        # trains all 5 models, saves pipelines + metrics.json
    ├── metrics.json           # evaluation metrics for all 5 models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest_ensemble.pkl
```

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (optional) retrain all models from scratch
python model/train_models.py

# launch the app
streamlit run app.py
```

## Streamlit App Features

- **Dataset upload (CSV):** upload your own test CSV (same schema as
  `test_data.csv`) or use the bundled sample.
- **Model selection dropdown:** choose any of the 5 trained models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC computed
  live on the uploaded data (when the true `Churn` column is present).
- **Confusion matrix & classification report:** visual confusion matrix
  heatmap plus a full `sklearn` classification report.
- A training-time comparison table across all 5 models is also shown for
  reference.

## Deployment

Deployed on Streamlit Community Cloud:

**Live App Link:** `<ADD_YOUR_STREAMLIT_APP_LINK_HERE_AFTER_DEPLOYING>`

Deployment steps:
1. Push this repository to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. Click "New app", select this repository/branch, and set the main file to `app.py`.
4. Click "Deploy".
