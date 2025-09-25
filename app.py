import streamlit as st
import pandas as pd
import pickle
import shap
import streamlit.components.v1 as components

# =====================
# Load trained pipeline
# =====================
with open("catboost_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("📊 Customer Churn Prediction with Explainability")
st.write("Enter customer details below:")

# =====================
# Collect raw user input
# =====================
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

# =====================
# Create dataframe from user input
# =====================
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

# =====================
# Run prediction
# =====================
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]

    st.subheader("🔮 Prediction Result")
    st.write(f"Churn: **{'Yes' if prediction == 1 else 'No'}**")
    st.write(f"Probability of Churn: **{proba:.2f}**")

    # =====================
    # SHAP Explainability
    # =====================
    st.subheader("📈 Model Explainability (SHAP)")

    model = pipeline.named_steps['catboost']
    preprocessor = pipeline.named_steps['preprocessor']

    # Transform input using preprocessor for SHAP
    input_transformed = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_transformed)

    # Generate SHAP force plot (HTML)
    shap_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        feature_names=feature_names,
        matplotlib=False
    )

    # Display in Streamlit without matplotlib
    components.html(shap_plot.html(), height=300)
