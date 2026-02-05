import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Student Performance Risk Prediction with Explainable AI (SHAP)")
st.caption("Upload student data to predict academic risk and understand key influencing factors")

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "rf_model.joblib"))
    encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.joblib"))
    return model, encoder

rf_model, label_encoder = load_artifacts()

# ================= FEATURES =================
FEATURES = [
    'attendance_pct',
    'quiz_1','quiz_2','quiz_3','quiz_4','quiz_5',
    'quiz_avg','quiz_std',
    'assignment_score',
    'sessional1','sessional2',
    'cheating_count','teacher_feedback_score'
]

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Student Data (CSV / Excel)",
    type=["csv", "xlsx"]
)

# ================= MAIN LOGIC =================
if uploaded_file is not None:

    # ---------- Read file ----------
    df = (
        pd.read_excel(uploaded_file)
        if uploaded_file.name.endswith(".xlsx")
        else pd.read_csv(uploaded_file)
    )

    # ---------- Feature engineering ----------
    quiz_cols = ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5']
    df['quiz_avg'] = df[quiz_cols].mean(axis=1)
    df['quiz_std'] = df[quiz_cols].std(axis=1)

    # ---------- Force numeric ----------
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # ---------- Predictions ----------
    preds = rf_model.predict(X)
    probs = rf_model.predict_proba(X)

    df['Predicted_Risk'] = label_encoder.inverse_transform(preds)

    for i, cls in enumerate(label_encoder.classes_):
        df[f'Prob_{cls}'] = probs[:, i]

    # ================= METRICS =================
    st.divider()
    col1, col2, col3 = st.columns(3)

    col1.metric("👨‍🎓 Total Students", len(df))
    col2.metric("⚠️ At Risk", (df["Predicted_Risk"] == "AtRisk").sum())
    col3.metric("🚨 Critical", (df["Predicted_Risk"] == "Critical").sum())

    # ================= RESULTS TABLE =================
    st.divider()
    st.subheader("📊 Prediction Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Prediction Results",
        data=df.to_csv(index=False),
        file_name="student_risk_predictions.csv",
        mime="text/csv"
    )

    # ================= RISK DISTRIBUTION =================
    st.divider()
    st.subheader("📈 Risk Distribution")

    risk_counts = df["Predicted_Risk"].value_counts()
    st.bar_chart(risk_counts, use_container_width=True)

    # ================= SHAP EXPLANATION =================
    st.divider()
    st.subheader("🧠 SHAP Explanation (Why this prediction?)")

    student_index = st.sidebar.selectbox(
        "Select Student Index",
        df.index
    )

    st.info(
        f"Explanation for predicted risk class: "
        f"**{df.loc[student_index, 'Predicted_Risk']}**"
    )

    # ---------- Background sample ----------
    background = X.sample(min(50, len(X)), random_state=42)

    def predict_proba_fn(data):
        data_df = pd.DataFrame(data, columns=FEATURES)
        return rf_model.predict_proba(data_df)

    explainer = shap.KernelExplainer(
        predict_proba_fn,
        background.values
    )

    shap_values = explainer.shap_values(
        X.iloc[[student_index]].values,
        silent=True
    )

    # ---------- Safe SHAP processing ----------
    shap_arr = np.abs(np.array(shap_values, dtype=object))
    shap_flat = np.concatenate([np.ravel(s) for s in shap_arr])

    if shap_flat.size >= len(FEATURES):
        shap_vector = shap_flat[:len(FEATURES)]
    else:
        shap_vector = np.pad(
            shap_flat,
            (0, len(FEATURES) - shap_flat.size),
            constant_values=0
        )

    shap_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Impact": shap_vector
        })
        .sort_values(by="Impact", ascending=False)
        .head(8)   # show top 8 only
    )

    st.dataframe(
        shap_df,
        use_container_width=True
    )

else:
    st.info("⬅️ Upload a CSV or Excel file from the sidebar to get started")
