import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Risk Prediction")
st.caption("Predict academic risk levels with explainable AI (SHAP)")

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
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This app predicts **student academic risk** using a  
        **Random Forest model** trained in Google Colab.

        It also explains *why* a prediction was made using **SHAP**.
        """
    )

# ================= FILE UPLOAD =================
st.header("📂 Upload Student Data")
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ---------- Read data ----------
    df = (
        pd.read_excel(uploaded_file)
        if uploaded_file.name.endswith(".xlsx")
        else pd.read_csv(uploaded_file)
    )

    st.subheader("🧾 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Feature engineering ----------
    quiz_cols = ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5']
    df['quiz_avg'] = df[quiz_cols].mean(axis=1)
    df['quiz_std'] = df[quiz_cols].std(axis=1)

    # ---------- Force numeric ----------
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    # ---------- Prediction ----------
    preds = rf_model.predict(X)
    probs = rf_model.predict_proba(X)

    df['Predicted_Risk'] = label_encoder.inverse_transform(preds)

    for i, cls in enumerate(label_encoder.classes_):
        df[f'Prob_{cls}'] = probs[:, i]

    # ---------- Results ----------
    st.divider()
    st.subheader("📊 Prediction Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Risk Distribution")
    st.bar_chart(df['Predicted_Risk'].value_counts())

    # ================= SHAP =================
    st.divider()
    st.subheader("🔍 Explain Prediction (SHAP)")

    student_index = st.selectbox(
        "Select Student Index",
        df.index
    )

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

    shap_arr = np.abs(np.array(shap_values, dtype=object))
    shap_flat = np.concatenate([np.ravel(s) for s in shap_arr])

    shap_vector = (
        shap_flat[:len(FEATURES)]
        if shap_flat.size >= len(FEATURES)
        else np.pad(shap_flat, (0, len(FEATURES) - shap_flat.size))
    )

    shap_df = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Impact": shap_vector
        })
        .sort_values("Impact", ascending=False)
    )

    st.success(
        f"🧠 Predicted Risk: **{df.loc[student_index, 'Predicted_Risk']}**"
    )

    st.dataframe(shap_df, use_container_width=True)
