import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ChurnGuard", layout="wide")

st.title("🔥 ChurnGuard - Customer Churn Prediction")
st.write("Upload customer data to predict churn")

@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

model = load_model()
st.success("✅ Model Loaded!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    probs = model.predict_proba(df)[:, 1]
    df['Churn_Probability'] = probs.round(4)
    df['Risk'] = pd.cut(probs, bins=[0, 0.5, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
    
    st.write("### Predictions")
    st.dataframe(df[['Churn_Probability', 'Risk']], use_container_width=True)
    
    high_risk = df[df['Risk'] == 'High']
    if not high_risk.empty:
        st.error(f"🚨 {len(high_risk)} High Risk Customers Found!")
        st.dataframe(high_risk, use_container_width=True)
    
    st.download_button("📥 Download Full Results", df.to_csv(index=False), "churn_predictions.csv", "text/csv")
