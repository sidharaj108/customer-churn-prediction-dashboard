# app.py 


import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ChurnGuard - AI Retention System", layout="wide")
st.title("🔥 ChurnGuard: Predict & Prevent Customer Churn")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

model = load_model()

# Sidebar
st.sidebar.header("Upload New Customers")
uploaded_file = st.sidebar.file_uploader("Upload CSV (same format as Telco data)", type="csv")

if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df_new.head())
    
    # Predict
    probabilities = model.predict_proba(df_new)[:,1]
    df_new['Churn_Probability'] = probabilities
    df_new['Churn_Risk'] = np.where(probabilities > 0.7, "High", 
                                   np.where(probabilities > 0.5, "Medium", "Low"))
    
    st.write("### Predictions")
    st.dataframe(df_new[['Churn_Probability', 'Churn_Risk']])
    
    # Top high-risk
    high_risk = df_new[df_new['Churn_Probability'] > 0.7].sort_values('Churn_Probability', ascending=False)
    st.write(f"### 🚨 Top {len(high_risk)} High-Risk Customers")
    st.dataframe(high_risk)
    
    # SHAP for first customer
    st.write("### SHAP Explanation (Why this customer is at risk?)")
    explainer = shap.TreeExplainer(model.named_steps['model'])
    sample = model.named_steps['preprocessor'].transform(df_new.iloc[[0]])
    shap_values = explainer.shap_values(sample)
    fig = shap.force_plot(explainer.expected_value, shap_values[0], df_new.iloc[0], matplotlib=True, show=False)
    st.pyplot(fig)

else:
    st.info("Upload a CSV or use the demo below 👇")

# Manual single prediction
st.subheader("Single Customer Prediction")
with st.form("single_pred"):
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        senior = st.selectbox("Senior Citizen?", ["Yes", "No"])
    
    submitted = st.form_submit_button("Predict Churn")
    if submitted:
        input_data = pd.DataFrame({
            'tenure': [tenure], 'MonthlyCharges': [monthly_charges],
            'Contract': [contract], 'InternetService': [internet],
            'PaymentMethod': [payment], 'SeniorCitizen': [1 if senior=="Yes" else 0],
            # Add other columns with median/default values (full code has all)
        })
        prob = model.predict_proba(input_data)[:,1][0]
        st.success(f"Churn Probability: **{prob:.1%}**")
        st.progress(float(prob))
