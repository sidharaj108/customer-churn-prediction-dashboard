import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="ChurnGuard - Customer Churn Prediction",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 ChurnGuard: AI Customer Retention Dashboard")
st.markdown("### Predict which customers are likely to churn and take action")

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_model.pkl')
        st.success("✅ Model Loaded Successfully")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("Please make sure 'churn_model.pkl' is in the same folder and redeploy.")
        return None

model = load_model()

if model is None:
    st.stop()

# ====================== SIDEBAR ======================
st.sidebar.header("📤 Upload Customer Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (same format as Telco data)", 
    type=["csv"]
)

# ====================== MAIN APP ======================
tab1, tab2 = st.tabs(["📊 Batch Prediction", "👤 Single Customer Prediction"])

# ====================== TAB 1: BATCH PREDICTION ======================
with tab1:
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df_new.head(), use_container_width=True)
        
        # Make Predictions
        probabilities = model.predict_proba(df_new)[:, 1]
        predictions = model.predict(df_new)
        
        df_new['Churn_Probability'] = probabilities
        df_new['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
        df_new['Risk_Level'] = pd.cut(probabilities, 
                                    bins=[0, 0.5, 0.7, 1.0], 
                                    labels=['Low', 'Medium', 'High'])
        
        # Display Results
        st.write("### Prediction Results")
        st.dataframe(df_new[['Churn_Probability', 'Churn_Prediction', 'Risk_Level']], 
                    use_container_width=True)
        
        # High Risk Customers
        high_risk = df_new[df_new['Risk_Level'] == 'High'].sort_values('Churn_Probability', ascending=False)
        st.write(f"### 🚨 High Risk Customers ({len(high_risk)})")
        st.dataframe(high_risk, use_container_width=True)
        
        # Download Button
        csv = df_new.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Upload a CSV file in the sidebar to get batch predictions")

# ====================== TAB 2: SINGLE PREDICTION ======================
with tab2:
    st.subheader("Manual Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
    
    with col2:
        multiplelines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paymentmethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthlycharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        totalcharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

    if st.button("🔮 Predict Churn", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if senior == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phoneservice],
            'MultipleLines': [multiplelines],
            'InternetService': [internetservice],
            'OnlineSecurity': ['No'],      # Default values (you can expand)
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': [contract],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': [paymentmethod],
            'MonthlyCharges': [monthlycharges],
            'TotalCharges': [totalcharges]
        })
        
        prob = model.predict_proba(input_data)[0][1]
        churn_pred = "Yes" if prob > 0.5 else "No"
        
        # Display Result
        st.success(f"**Churn Probability: {prob:.1%}**")
        
        if prob > 0.7:
            st.error("🚨 HIGH RISK - Customer is likely to churn")
        elif prob > 0.5:
            st.warning("⚠️ MEDIUM RISK")
        else:
            st.success("✅ LOW RISK")
        
        # Recommendation
        if prob > 0.7:
            st.info("**Recommended Action**: Offer 20% discount + Free upgrade + Immediate call")
        elif prob > 0.5:
            st.info("**Recommended Action**: Send personalized retention email + Loyalty points")
        
        st.progress(float(prob))

# Footer
st.caption("ChurnGuard - Built for Customer Retention | Powered by XGBoost + SHAP")
