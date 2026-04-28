"""
Predict churn probability for a new customer.
Run after training the model:
    python src/train_model.py
    python src/predict.py
"""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def predict_customer(customer_data: dict):
    """Return churn prediction and churn probability for one customer."""
    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    return prediction, probability


if __name__ == "__main__":
    sample_customer = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 50000,
    }

    prediction, probability = predict_customer(sample_customer)

    print("Sample Customer Churn Prediction")
    print("--------------------------------")
    print(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
    print(f"Churn Probability: {probability:.2%}")
