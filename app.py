import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Student Risk Analytics",
    page_icon="🎓",
    layout="wide"
)

# ---------------- Load Assets ----------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.joblib")

model = load_model()
label_encoder = load_encoder()

# ---------------- Sidebar ----------------
st.sidebar.title("🎛 Control Panel")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Student Data (Excel)",
    type=["xlsx"]
)

# ---------------- Header ----------------
st.markdown(
    """
    <h1 style='text-align:center;'>🎓 Student Risk Prediction Dashboard</h1>
    <p style='text-align:center; color:gray;'>AI-powered academic risk analysis</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- Main Logic ----------------
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    X = df.copy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X).max(axis=1) * 100

    df["Predicted_Risk"] = label_encoder.inverse_transform(predictions)
    df["Confidence (%)"] = probabilities.round(2)

    # ---------- Prediction Results ----------
    st.subheader("📊 Prediction Results")
    st.dataframe(df, use_container_width=True, height=400)

    # ---------- Risk Distribution ----------
    st.subheader("📈 Risk Distribution")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.bar_chart(df["Predicted_Risk"].value_counts())

    with col2:
        st.metric("Total Students", len(df))
        st.metric("At Risk", (df["Predicted_Risk"] == "AtRisk").sum())
        st.metric("Safe", (df["Predicted_Risk"] == "Good").sum())

    st.markdown("---")

    # ---------- SHAP Section ----------
    st.subheader("🧠 Model Insights")

    student_index = st.slider(
        "Select Student Index",
        0,
        len(df) - 1,
        0
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    predicted_class = predictions[student_index]
    class_name = label_encoder.inverse_transform([predicted_class])[0]

    st.markdown(
        f"""
        <div style="padding:15px; border-radius:10px; background-color:#f5f7fa;">
            <h3 style="margin-bottom:0;">Prediction</h3>
            <h2 style="color:#1f77b4;">{class_name}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "Impact": shap_values[predicted_class][student_index]
    }).sort_values(by="Impact", ascending=False)

    st.markdown("### 🔍 Feature Impact")
    st.dataframe(shap_df, use_container_width=True, height=350)

    st.markdown("---")

    st.success("✔ Analysis Complete")

else:
    st.markdown(
        """
        <div style="text-align:center; padding:60px;">
            <h2>📂 Upload Student Dataset</h2>
            <p style="color:gray;">Use the sidebar to begin analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )
