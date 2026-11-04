"""Day 4 feature engineering for the RetailMind Telco Churn project."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add deterministic domain features before column preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

        if "tenure" in X.columns:
            X["tenure_bucket"] = pd.cut(
                X["tenure"],
                bins=[-1, 12, 24, 48, 60, np.inf],
                labels=["0-12", "13-24", "25-48", "49-60", "61+"]
            ).astype("object")
            X["is_new_customer"] = (X["tenure"] <= 12).astype(int)
            X["is_long_term_customer"] = (X["tenure"] >= 48).astype(int)

        if {"TotalCharges", "tenure"}.issubset(X.columns):
            X["avg_charge_per_tenure"] = X["TotalCharges"] / X["tenure"].replace(0, 1)

        if {"MonthlyCharges", "TotalCharges"}.issubset(X.columns):
            X["monthly_to_total_ratio"] = (
                X["MonthlyCharges"] / X["TotalCharges"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)

        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        available_services = [c for c in service_cols if c in X.columns]
        if available_services:
            X["num_services"] = sum(
                X[c].astype(str).str.lower().eq("yes").astype(int)
                for c in available_services
            )

        protection_cols = [c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"] if c in X.columns]
        if protection_cols:
            X["has_protection"] = (
                sum(X[c].astype(str).str.lower().eq("yes").astype(int) for c in protection_cols) > 0
            ).astype(int)

        if "Contract" in X.columns:
            X["is_month_to_month"] = X["Contract"].astype(str).eq("Month-to-month").astype(int)
            X["has_long_contract"] = X["Contract"].astype(str).isin(["One year", "Two year"]).astype(int)

        if "PaymentMethod" in X.columns:
            X["uses_electronic_check"] = X["PaymentMethod"].astype(str).eq("Electronic check").astype(int)

        if "PaperlessBilling" in X.columns:
            X["paperless_flag"] = X["PaperlessBilling"].astype(str).str.lower().eq("yes").astype(int)

        return X
