# ============================================================
# Customer Churn Prediction & Sales Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. LOAD DATA
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction & Sales Dashboard")

# Load datasets
@st.cache_data
def load_data():
    customers = pd.read_csv("customer_data.csv")
    transactions = pd.read_csv("transaction_data.csv")
    return customers, transactions

customers, transactions = load_data()

# Convert date column
transactions["date"] = pd.to_datetime(transactions["date"])

# -------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------
agg_trans = transactions.groupby("customer_id").agg({
    "amount": ["sum", "mean", "count"],
    "date": ["min", "max"]
})
agg_trans.columns = ["_".join(x) for x in agg_trans.columns.ravel()]
agg_trans["days_active"] = (agg_trans["date_max"] - agg_trans["date_min"]).dt.days

data = customers.merge(agg_trans, left_on="customer_id", right_index=True, how="left").fillna(0)

# Prepare model data
X = data.drop(columns=["customer_id", "name", "churn", "date_min", "date_max"])
y = data["churn"]
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 3. DASHBOARD LAYOUT
# -------------------------------
tab1, tab2 = st.tabs(["📈 Sales Trend", "🔮 Churn Prediction"])

# -------------------------------
# TAB 1 – SALES TREND
# -------------------------------
with tab1:
    st.subheader("Monthly Sales Trend")

    transactions["month"] = transactions["date"].dt.to_period("M")
    monthly_sales = transactions.groupby("month")["amount"].sum().reset_index()
    monthly_sales["month"] = monthly_sales["month"].dt.to_timestamp()

    st.line_chart(monthly_sales.set_index("month"))

    # Top 5 Products
    st.subheader("Top 5 Performing Products")
    top_products = transactions.groupby("product")["amount"].sum().nlargest(5)
    st.bar_chart(top_products)

# -------------------------------
# TAB 2 – CHURN PREDICTION
# -------------------------------
with tab2:
    st.subheader("Predict Customer Churn")

    customer_id = st.number_input("Enter Customer ID:", min_value=1, max_value=int(data["customer_id"].max()))

    if st.button("🔍 Predict Churn"):
        if customer_id in list(data["customer_id"]):
            input_data = data[data["customer_id"] == customer_id].drop(columns=["customer_id","name","churn","date_min","date_max"])
            input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=X.columns, fill_value=0)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)[0][1]

            st.write(f"**Prediction:** {'⚠️ Churn' if prediction[0]==1 else '✅ Not Churn'}")
            st.progress(int(prob*100))
            st.caption(f"Churn Probability: {prob*100:.2f}%")
        else:
            st.error("Customer ID not found. Please enter a valid ID.")
