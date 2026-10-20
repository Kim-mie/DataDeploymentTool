"""Train EVERY model, evaluate them all the same way, and save them for the app.

Run from the repo root:  python train_all.py
Creates the folder  models/  with:
  <model>.joblib            one full pipeline per classical model
  preprocessor.joblib       encoder+scaler used by the neural net
  deep_neural_net.keras     the neural net
  meta.json                 input columns + test metrics for every model
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
tf.keras.utils.set_random_seed(SEED)
URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "refs/heads/master/data/Telco-Customer-Churn.csv")
OUT = "models"
os.makedirs(OUT, exist_ok=True)

# ---------- data (same cleaning as the notebook) ----------
df = pd.read_csv(URL).drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X, y = df.drop(columns=["Churn"]), df["Churn"]

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT = [c for c in X.columns if c not in NUM]


def make_pre():
    # scaling is harmless for trees and required for LR / neural net
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
neg, pos = np.bincount(y_train)

# ---------- classical models (all class-balanced) ----------
classical = {
    "decision_tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=10, criterion="entropy",
        class_weight="balanced", random_state=SEED),
    "logistic_regression": LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=SEED),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=neg / pos, eval_metric="logloss", random_state=SEED),
}


def score(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),   # on probabilities
    }


results = {}
for name, clf in classical.items():
    pipe = Pipeline([("prep", make_pre()), ("clf", clf)]).fit(X_train, y_train)
    results[name] = score(y_test, pipe.predict_proba(X_test)[:, 1])
    joblib.dump(pipe, f"{OUT}/{name}.joblib")
    print(name, results[name])

# ---------- neural net (needs scaled input -> separate preprocessor) ----------
pre = make_pre().fit(X_train)
Xtr, Xte = pre.transform(X_train), pre.transform(X_test)
joblib.dump(pre, f"{OUT}/preprocessor.joblib")

dnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
dnn.compile(optimizer="adam", loss="binary_crossentropy")
dnn.fit(
    Xtr, y_train.values, validation_split=0.15, epochs=200, batch_size=32,
    class_weight={0: 1.0, 1: neg / pos}, verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)])
results["deep_neural_net"] = score(y_test, dnn.predict(Xte, verbose=0).ravel())
dnn.save(f"{OUT}/deep_neural_net.keras")
print("deep_neural_net", results["deep_neural_net"])

with open(f"{OUT}/meta.json", "w") as f:
    json.dump({"columns": list(X.columns), "metrics": results}, f, indent=2)

print("\n" + pd.DataFrame(results).T.sort_values("roc_auc", ascending=False).to_string())
