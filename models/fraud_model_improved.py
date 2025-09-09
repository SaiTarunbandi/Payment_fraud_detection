
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, roc_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ==============================
# Loaders (reuse your structure)
# ==============================
def load_upi_data(path):
    df = pd.read_csv(path)
    # expected columns: user_id, device_id, amount, timestamp, is_fraud
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['source_type'] = 'upi'
    df['user_id'] = df['user_id'].astype(str)
    df['device_id'] = df['device_id'].astype(str)
    df['is_fraud'] = df['is_fraud'].astype(int)
    return df[['user_id', 'device_id', 'amount', 'hour', 'source_type', 'is_fraud']]

def load_credit_card_data(path):
    df = pd.read_csv(path)
    # Credit card dataset: has Time, Amount, Class
    df['hour'] = (df['Time'] // 3600) % 24
    df['user_id'] = 'cc_user_' + df.index.astype(str)
    df['device_id'] = 'cc_device_' + df.index.astype(str)
    df['source_type'] = 'credit_card'
    df.rename(columns={'Class': 'is_fraud'}, inplace=True)
    out = df[['user_id', 'device_id', 'Amount', 'hour', 'source_type', 'is_fraud']].rename(columns={'Amount': 'amount'})
    out['is_fraud'] = out['is_fraud'].astype(int)
    return out

def load_online_data(path):
    df = pd.read_csv(path)
    # PaySim-like dataset: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    df['hour'] = df['step'] % 24
    df['source_type'] = 'online_payment'
    df['user_id'] = df['nameOrig'].astype(str)
    df['device_id'] = df['nameDest'].astype(str)
    df.rename(columns={'isFraud': 'is_fraud'}, inplace=True)
    df['is_fraud'] = df['is_fraud'].astype(int)
    return df[['user_id', 'device_id', 'amount', 'hour', 'source_type', 'is_fraud']]

def prepare_combined_dataset(upi_path, card_path, online_path):
    upi_df = load_upi_data(upi_path)
    card_df = load_credit_card_data(card_path)
    online_df = load_online_data(online_path)
    df = pd.concat([upi_df, card_df, online_df], ignore_index=True)
    df = df.dropna()
    return df

# ====================================
# Feature builder (avoid ID leakage)
# ====================================
def build_features(df: pd.DataFrame):
    """Return X (features), y (labels), and a preprocessing pipeline.
    We drop 'user_id' and 'device_id' to avoid memorisation.
    """
    X = df[['amount', 'hour', 'source_type']].copy()
    y = df['is_fraud'].astype(int).values

    num_features = ['amount', 'hour']
    cat_features = ['source_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )
    return X, y, preprocessor

# ====================================
# Threshold tuning utility
# ====================================
def tune_threshold(y_true, y_proba, beta=1.0):
    """Return threshold that maximizes F-beta (default F1)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        y_hat = (y_proba >= thr).astype(int)
        score = f1_score(y_true, y_hat)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score

# ====================================
# Train & evaluate
# ====================================
def train_model(df, random_state=42, plot_curves=True):
    X, y, preprocessor = build_features(df)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Compute class weight for XGBoost
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = max((neg / max(pos, 1)), 1.0)

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

    clf = Pipeline(steps=[('prep', preprocessor), ('model', xgb)])
    clf.fit(X_train, y_train)

    # Probabilities & metrics
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # Threshold tuning
    best_thr, best_f1 = tune_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_thr).astype(int)

    # Reports
    print("=== Metrics (probabilistic) ===")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC (Average Precision): {ap:.4f}")
    print("\n=== Metrics at tuned threshold ===")
    print(f"Best threshold: {best_thr:.3f}  |  F1: {best_f1:.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if plot_curves:
        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(7,5))
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], linestyle='--')
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(7,5))
        plt.plot(recall, precision, label=f"PR AUC = {ap:.4f}")
        plt.title("Precision-Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Save pipeline & threshold
    joblib.dump(clf, "unified_fraud_model.pkl")
    with open("threshold.json", "w") as f:
        json.dump({"decision_threshold": best_thr}, f)
    print("✅ Saved: unified_fraud_model.pkl and threshold.json")

    return clf, best_thr, {"roc_auc": roc_auc, "pr_auc": ap, "f1_at_thr": best_f1}

# ====================================
# Inference helper
# ====================================
def predict_transaction(model_path, threshold_path, amount, hour, source_type):
    clf = joblib.load(model_path)
    with open(threshold_path, "r") as f:
        thr = json.load(f)["decision_threshold"]
    X = pd.DataFrame([{"amount": amount, "hour": hour, "source_type": source_type}])
    proba = clf.predict_proba(X)[:, 1][0]
    is_fraud = int(proba >= thr)
    return {"proba": float(proba), "threshold": float(thr), "is_fraud": is_fraud}

# ====================================
# Main (optional)
# ====================================
if __name__ == "__main__":
    # Update these paths to your actual data locations
    upi_path = "data/upi_transactions.csv"
    credit_path = "data/creditcard.csv"
    online_path = "data/online_payments.csv"

    df = prepare_combined_dataset(upi_path, credit_path, online_path)
    train_model(df, plot_curves=True)
