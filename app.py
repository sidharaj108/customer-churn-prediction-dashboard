import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="ChurnGuard", layout="wide")

st.title("🔥 ChurnGuard - Customer Churn Prediction")
st.markdown("### Predict & Prevent Customer Churn")

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('churn_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

st.success("✅ Model Loaded Successfully!")

# File Upload
uploaded_file = st.file_uploader("Upload Customer CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    probs = model.predict_proba(df)[:, 1]
    df['Churn_Probability'] = probs
    df['Risk'] = pd.cut(probs, bins=[0, 0.5, 0.7, 1], labels=['Low', 'Medium', 'High'])
    
    st.write("### Prediction Results")
    st.dataframe(df[['Churn_Probability', 'Risk']], use_container_width=True)
    
    high_risk = df[df['Risk'] == 'High']
    st.write(f"### 🚨 High Risk Customers ({len(high_risk)})")
    st.dataframe(high_risk, use_container_width=True)
    
    # Download
    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
