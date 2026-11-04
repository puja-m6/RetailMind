"""Train and persist the Day 4 RetailMind churn champion."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import TelcoFeatureEngineer

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
DATA_URL="https://raw.githubusercontent.com/Pavan-gs/ai-services-bbsr/refs/heads/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(url=DATA_URL):
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    return X, y


def make_preprocessor(X_train):
    sample = TelcoFeatureEngineer().fit_transform(X_train)
    numeric_cols = sample.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = sample.select_dtypes(exclude=np.number).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder)
        ]), categorical_cols)
    ])


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }


def build_models(preprocessor):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=350, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    return {
        name: Pipeline([
            ("features", TelcoFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        for name, estimator in models.items()
    }


def train_and_save():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocessor = make_preprocessor(X_train)
    models = build_models(preprocessor)

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        fitted[name] = model

    champion_name = max(results, key=lambda name: (results[name]["roc_auc"], results[name]["f1"]))
    champion = fitted[champion_name]

    joblib.dump(champion, MODEL_DIR / "churn_champion_day4.joblib")
    with open(MODEL_DIR / "day4_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"champion": champion_name, "models": results}, f, indent=2)

    print(pd.DataFrame(results).T.sort_values("roc_auc", ascending=False).round(4))
    print(f"\nSaved champion: {champion_name}")
    print(MODEL_DIR / "churn_champion_day4.joblib")
    return champion_name, results


if __name__ == "__main__":
    train_and_save()
