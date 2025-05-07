import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Sample Data (Replace with yours)
# ------------------------------
models = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBoost']
metrics = {
    "Accuracy": [0.5556, 0.8293, 0.7913, 0.7696],
    "Precision": [0.5564, 0.8320, 0.7913, 0.7700],
    "Recall": [0.5559, 0.8296, 0.7913, 0.7698],
    "F1 Score": [0.5548, 0.8290, 0.7913, 0.7696]
}
df_metrics = pd.DataFrame(metrics, index=models)

# ------------------------------
# Streamlit Layout
# ------------------------------
st.set_page_config(page_title="Telecom Churn Dashboard", layout="wide")

st.title("📊 Telecom Churn Prediction Dashboard")
st.markdown("A machine learning project to identify and reduce customer churn in Ghana's telecom industry.")

# Metrics Overview
st.subheader("📌 Model Performance Comparison")
st.dataframe(df_metrics.style.highlight_max(axis=0), height=300)

# Confusion Matrices (Dummy for now – replace with actual)
st.subheader("🔎 Confusion Matrices")
cols = st.columns(2)
for i, model in enumerate(models[:2]):
    with cols[i]:
        st.markdown(f"**{model}**")
        fig, ax = plt.subplots()
        sns.heatmap([[100, 30], [40, 90]], annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

cols = st.columns(2)
for i, model in enumerate(models[2:]):
    with cols[i]:
        st.markdown(f"**{model}**")
        fig, ax = plt.subplots()
        sns.heatmap([[110, 20], [50, 80]], annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# Business Impact
st.subheader("📉 Churn Impact Analysis")
st.markdown("""
- **MTN**: Largest subscriber base; high churn impacts market dominance.
- **Vodafone**: Significant churn tied to pricing dissatisfaction.
- **AirtelTigo**: Network issues were the leading cause of churn.
- **Glo**: Lowest base but highest relative churn percentage.
""")

# Recommendations
st.subheader("✅ Recommendations")
st.markdown("""
1. **Personalized Offers**: Tailored retention strategies.
2. **Improve Network Quality**: Especially AirtelTigo and Glo.
3. **Customer Support Training**: To reduce service-related churn.
4. **Pricing Optimization**: For Vodafone.
5. **Churn Alert System**: Deploy best model (Random Forest) in production.
""")

# Downloads
st.subheader("⬇️ Downloads")
st.download_button("📥 Download Full Report (PDF)", open("report.pdf", "rb").read(), file_name="Churn_Report.pdf")
st.download_button("📥 Download Model", open("random_forest_model.pkl", "rb").read(), file_name="best_model.pkl")

# Optional: Live Prediction Interface
st.subheader("🧠 Try a Prediction")
monthly_charges = st.slider("Monthly Charges", 0, 500, 100)
duration = st.slider("Duration with Company (months)", 0, 60, 12)
company = st.selectbox("Telecom Company", ['MTN', 'Vodafone', 'AirtelTigo', 'Glo'])

if st.button("Predict Churn Likelihood"):
    # Dummy prediction
    st.success("Customer has a **low churn risk**.")

