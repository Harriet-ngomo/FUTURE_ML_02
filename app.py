import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components  # ✅ needed for force plot

# =====================
# Helper to render SHAP force plots
# =====================
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# =====================
# Page Config
# =====================
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# =====================
# Load trained pipeline safely
# =====================
if not os.path.exists("catboost_pipeline.pkl"):
    st.error("❌ Model file (catboost_pipeline.pkl) not found. Please upload it to the repository root.")
    st.stop()

with open("catboost_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# =====================
# Title
# =====================
st.title("📊 Customer Churn Prediction with Explainability")

# =====================
# Two-column layout
# =====================
col1, col2 = st.columns([1, 2])

# ---------------------
# LEFT COLUMN (Intro + Inputs)
# ---------------------
with col1:
    st.header("👋 Welcome!")
    st.markdown("""
    This interactive app predicts **customer churn** using a trained CatBoost model.  

    - Enter customer details below  
    - Get a **prediction** with churn probability  
    - Explore **explainability plots (SHAP)** showing *why* the prediction was made  
    """)

    st.subheader("Enter Customer Details")

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

    # Prepare input dataframe
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

    # Derived features
    input_data['AvgMonthlySpend'] = input_data['TotalCharges'] / (input_data['tenure'] + 1)
    input_data['NumServices'] = (
        (input_data['PhoneService'] == 'Yes').astype(int) +
        (input_data['InternetService'] != 'No').astype(int) +
        (input_data['OnlineSecurity'] == 'Yes').astype(int) +
        (input_data['OnlineBackup'] == 'Yes').astype(int) +
        (input_data['DeviceProtection'] == 'Yes').astype(int) +
        (input_data['TechSupport'] == 'Yes').astype(int) +
        (input_data['StreamingTV'] == 'Yes').astype(int) +
        (input_data['StreamingMovies'] == 'Yes').astype(int)
    )
    input_data['IsSenior'] = input_data['SeniorCitizen']
    input_data['TenureGroup'] = pd.cut(
        input_data['tenure'],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=['0-12','13-24','25-48','49-60','61-72']
    )
    input_data['PaymentTypeSimple'] = input_data['PaymentMethod'].apply(
        lambda x: 'Electronic' if 'Electronic' in x else ('Automatic' if 'automatic' in x else 'Mailed')
    )

# ---------------------
# RIGHT COLUMN (Prediction + Explainability)
# ---------------------
with col2:
    if st.button("🚀 Predict Churn"):
        try:
            prediction = pipeline.predict(input_data)[0]
            proba = pipeline.predict_proba(input_data)[0][1]

            st.subheader("🔮 Prediction Result")
            st.write(f"Churn: **{'Yes' if prediction == 1 else 'No'}**")
            st.write(f"Probability of Churn: **{proba:.2f}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # =====================
        # Churn Probability Distribution
        # =====================
        st.subheader("📊 Churn Probability Distribution")
        try:
            if hasattr(pipeline, "X_train_"):
                all_probs = pipeline.predict_proba(pipeline.X_train_)[:, 1]
                fig, ax = plt.subplots()
                sns.histplot(all_probs, bins=20, kde=True, ax=ax)
                ax.axvline(proba, color="red", linestyle="--", label="Current Customer")
                ax.set_title("Distribution of Predicted Churn Probabilities")
                ax.set_xlabel("Churn Probability")
                ax.set_ylabel("Count")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("ℹ️ Training data not saved in pipeline, skipping distribution plot.")
        except Exception as e:
            st.warning(f"Could not generate distribution plot: {e}")

        # =====================
        # SHAP Explainability
        # =====================
        st.subheader("📈 Model Explainability (SHAP)")

        try:
            model = pipeline.named_steps['catboost']
            preprocessor = pipeline.named_steps['preprocessor']

            # Local explanation
            input_transformed = preprocessor.transform(input_data)
            feature_names = preprocessor.get_feature_names_out()
            explainer = shap.TreeExplainer(model)
            shap_values_input = explainer(input_transformed)

            st.markdown("#### 🔍 Why this prediction?")

            # --- Interactive Force Plot ---
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values_input.values[0],
                feature_names=feature_names,
                matplotlib=False
            )
            st_shap(force_plot, height=300)

            # --- Static Waterfall as backup ---
            fig_waterfall, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values_input[0], show=False)
            st.pyplot(fig_waterfall)

            # Global explanation (using training set if available)
            st.markdown("#### 🌍 Global Feature Importance")
            if hasattr(pipeline, "X_train_"):
                X_train_transformed = preprocessor.transform(pipeline.X_train_)
                shap_values_global = explainer(X_train_transformed)

                fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values_global.values,
                    X_train_transformed,
                    feature_names=feature_names,
                    show=False
                )
                st.pyplot(fig_summary)
            else:
                st.info("ℹ️ Training data not available in pipeline. Showing only single-customer explanation.")

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
