import streamlit as st
import joblib
import pandas as pd

# Load trained model and encoders
model = joblib.load("model/churn_model.joblib")
encoders = joblib.load("model/encoders.joblib")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])

if senior == "Yes":
    senior = 1
else:
    senior = 0
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

phone = st.selectbox("Phone Service", ["No", "Yes"])
multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -------------------------
# Encode categorical values
# -------------------------


monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device],
        "TechSupport": [tech],
        "StreamingTV": [tv],
        "StreamingMovies": [movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    # Encode categorical columns
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col].astype(str))

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")
    else:
        st.success("✅ Customer is likely to Stay")
        st.metric("Prediction Confidence", f"{confidence:.2f}%")

st.divider()

if st.button("🔄 Reset Form"):
    st.rerun()