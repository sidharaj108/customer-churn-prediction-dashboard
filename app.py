# ============================================================
# ADVANCED CUSTOMER CHURN PREDICTION & SALES DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from prophet import Prophet

# -------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📈 Customer Churn Prediction & Sales Analytics Dashboard")
st.markdown("Analyze sales trends, forecast growth, and predict customer churn in real-time.")

# -------------------------------
# 2. LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    customers = pd.read_csv("data/customer_data.csv")
    transactions = pd.read_csv("data/transaction_data.csv")
    transactions["date"] = pd.to_datetime(transactions["date"])
    return customers, transactions

customers, transactions = load_data()

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
agg_trans = transactions.groupby("customer_id").agg({
    "amount": ["sum", "mean", "count"],
    "date": ["min", "max"]
})
agg_trans.columns = ["_".join(x) for x in agg_trans.columns.ravel()]
agg_trans["days_active"] = (agg_trans["date_max"] - agg_trans["date_min"]).dt.days

data = customers.merge(agg_trans, left_on="customer_id", right_index=True, how="left").fillna(0)

X = data.drop(columns=["customer_id", "name", "churn", "date_min", "date_max"])
y = data["churn"]

X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 4. SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Dashboard Filters")

# Product filter
product_list = transactions["product"].unique().tolist()
selected_product = st.sidebar.selectbox("Select Product:", ["All"] + product_list)

# Date range filter
min_date, max_date = transactions["date"].min(), transactions["date"].max()
date_range = st.sidebar.date_input("Select Date Range:", [min_date, max_date])

# Apply filters
filtered_txn = transactions.copy()
if selected_product != "All":
    filtered_txn = filtered_txn[filtered_txn["product"] == selected_product]
filtered_txn = filtered_txn[(filtered_txn["date"] >= pd.Timestamp(date_range[0])) &
                            (filtered_txn["date"] <= pd.Timestamp(date_range[1]))]

# -------------------------------
# 5. DASHBOARD TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sales Overview", 
    "🔮 Churn Prediction", 
    "👥 Customer Insights",
    "📈 Advanced Analytics"
])

# ============================================================
# TAB 1 - SALES OVERVIEW
# ============================================================
with tab1:
    st.subheader("💰 Sales Trend & Forecast")

    col1, col2, col3 = st.columns(3)
    total_sales = filtered_txn["amount"].sum()
    total_orders = len(filtered_txn)
    avg_order = filtered_txn["amount"].mean()

    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Average Order Value", f"${avg_order:,.2f}")

    # Monthly trend
    filtered_txn["month"] = filtered_txn["date"].dt.to_period("M")
    monthly_sales = filtered_txn.groupby("month")["amount"].sum().reset_index()
    monthly_sales["month"] = monthly_sales["month"].dt.to_timestamp()
    st.line_chart(monthly_sales.set_index("month"))

    # Top 10 products
    st.subheader("🏆 Top Performing Products")
    top_products = filtered_txn.groupby("product")["amount"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    # Prophet Forecast
    st.subheader("🔮 Sales Forecast (Next 6 Months)")
    forecast_df = monthly_sales.rename(columns={"month": "ds", "amount": "y"})
    if len(forecast_df) >= 2:
        prophet = Prophet()
        prophet.fit(forecast_df)
        future = prophet.make_future_dataframe(periods=6, freq="M")
        forecast = prophet.predict(future)
        fig1 = prophet.plot(forecast)
        st.pyplot(fig1)

# ============================================================
# TAB 2 - CHURN PREDICTION
# ============================================================
with tab2:
    st.subheader("🔍 Predict Customer Churn")

    customer_id = st.number_input("Enter Customer ID:", min_value=1, max_value=int(data["customer_id"].max()))
    if st.button("Predict Churn"):
        input_data = data[data["customer_id"] == customer_id].drop(columns=["customer_id","name","churn","date_min","date_max"])
        input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=X.columns, fill_value=0)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown(f"### Prediction: {'⚠️ Churn' if prediction[0]==1 else '✅ Not Churn'}")
        st.progress(int(prob * 100))
        st.caption(f"Churn Probability: **{prob*100:.2f}%**")

# ============================================================
# TAB 3 - CUSTOMER INSIGHTS
# ============================================================
with tab3:
    st.subheader("👥 Customer Segmentation & Revenue Insights")

    col1, col2 = st.columns(2)
    churn_rate = data["churn"].mean() * 100
    churned_revenue = data[data["churn"]==1]["amount_sum"].sum()
    retained_revenue = data[data["churn"]==0]["amount_sum"].sum()

    with col1:
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

    with col2:
        fig, ax = plt.subplots()
        ax.pie([churned_revenue, retained_revenue],
               labels=["Churned Customers","Retained Customers"],
               autopct="%1.1f%%", colors=["#ff4b4b","#4bb543"])
        ax.set_title("Revenue Distribution")
        st.pyplot(fig)

# ============================================================
# TAB 4 - ADVANCED ANALYTICS
# ============================================================
with tab4:
    st.subheader("📊 Advanced Analytics & Charts")

    # 1️⃣ Distribution of Spend by Churn
    st.markdown("### Distribution of Total Spend by Churn Status")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=data, x="amount_sum", hue="churn", kde=True, bins=30, ax=ax)
    ax.set_xlabel("Total Spend per Customer")
    st.pyplot(fig)

    # 2️⃣ Boxplot: Days Active by Churn
    st.markdown("### Days Active by Churn Status")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x="churn", y="days_active", data=data, palette="Set2", ax=ax)
    ax.set_xticklabels(["Not Churned","Churned"])
    st.pyplot(fig)

    # 3️⃣ Subscription Type vs Churn
    if "subscription_type" in data.columns:
        st.markdown("### Subscription Type vs Churn Rate")
        sub = data.groupby("subscription_type")["churn"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x="subscription_type", y="churn", data=sub, ax=ax)
        ax.set_ylabel("Churn Rate")
        st.pyplot(fig)

    # 4️⃣ Scatter Plot
    st.markdown("### Scatter: Amount Spent vs Days Active (colored by churn)")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=data["days_active"], y=data["amount_sum"], hue=data["churn"],
                    palette=["green","red"], alpha=0.7, ax=ax)
    ax.set_xlabel("Days Active")
    ax.set_ylabel("Total Spend")
    st.pyplot(fig)

    # 5️⃣ Cumulative Sales (Area Chart)
    st.markdown("### Cumulative Sales Over Time")
    transactions["month"] = transactions["date"].dt.to_period("M")
    ms = transactions.groupby("month")["amount"].sum().reset_index()
    ms["month"] = ms["month"].dt.to_timestamp()
    ms["cumulative"] = ms["amount"].cumsum()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.fill_between(ms["month"], ms["cumulative"], color="skyblue", alpha=0.4)
    ax.plot(ms["month"], ms["cumulative"], color="blue", linewidth=2)
    ax.set_ylabel("Cumulative Sales")
    st.pyplot(fig)
