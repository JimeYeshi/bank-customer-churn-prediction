"""
Streamlit app for bank customer churn prediction.

Run:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

st.set_page_config(page_title="Bank Customer Churn Prediction", layout="centered")

st.title("Bank Customer Churn Prediction")
st.write(
    "This app predicts whether a banking customer is likely to churn based on customer profile and account behavior."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

credit_score = st.slider("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure with Bank", 0, 10, 3)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0)

customer = pd.DataFrame(
    [
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
        }
    ]
)

if st.button("Predict Churn Risk"):
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")

    if prediction == 1:
        st.warning("This customer is predicted to be at risk of churn.")
    else:
        st.success("This customer is predicted to be less likely to churn.")

    st.write("Customer input used for prediction:")
    st.dataframe(customer)
