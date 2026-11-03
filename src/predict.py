import joblib
import pandas as pd


model = joblib.load("models/churn_pipeline.joblib")


def predict(customer):
    data = pd.DataFrame([customer])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0, 1]

    return {
        "churn": int(prediction),
        "probability": float(probability),
    }
