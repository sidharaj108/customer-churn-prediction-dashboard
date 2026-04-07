import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

st.set_page_config(page_title="ChurnGuard", layout="wide")
st.title("🔥 ChurnGuard - Customer Churn Prediction")

@st.cache_resource
def load_model():
    try:
        # Try loading full pipeline first
        model = joblib.load('churn_model.pkl')
        st.success("✅ Full Model Loaded")
        return model
    except:
        try:
            # Fallback: Load XGBoost JSON model
            model = xgb.XGBClassifier()
            model.load_model('xgb_model.json')
            st.success("✅ XGBoost Model Loaded")
            return model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.stop()

model = load_model()

# File Upload
uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview", df.head())
    
    probs = model.predict_proba(df)[:, 1]
    df['Churn_Probability'] = probs.round(4)
    df['Risk_Level'] = pd.cut(probs, [0, 0.5, 0.7, 1], labels=['Low', 'Medium', 'High'])
    
    st.write("### Predictions")
    st.dataframe(df[['Churn_Probability', 'Risk_Level']], use_container_width=True)
    
    high_risk = df[df['Risk_Level'] == 'High']
    if len(high_risk) > 0:
        st.error(f"🚨 {len(high_risk)} High Risk Customers")
        st.dataframe(high_risk, use_container_width=True)
    
    st.download_button("Download Results", df.to_csv(index=False), "churn_predictions.csv")
