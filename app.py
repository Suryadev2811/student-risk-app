import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Student Risk Predictor", layout="wide")
st.title("🎓 Student Performance Risk Prediction with Explainable AI (SHAP)")

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

N_FEATURES = len(FEATURES)

# ================= FILE UPLOAD =================
st.header("📂 Upload Student Data (CSV / Excel)")
uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ---------- Read data ----------
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    # ---------- Feature engineering ----------
    quiz_cols = ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5']
    df['quiz_avg'] = df[quiz_cols].mean(axis=1)
    df['quiz_std'] = df[quiz_cols].std(axis=1)

    # ---------- Force numeric ----------
    X = (
    df[FEATURES]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)


    # ---------- Prediction ----------
    preds = rf_model.predict(X)
    probs = rf_model.predict_proba(X)

    df['Predicted_Risk'] = label_encoder.inverse_transform(preds)

    for i, cls in enumerate(label_encoder.classes_):
        df[f'Prob_{cls}'] = probs[:, i]

    # ---------- Display ----------
    st.subheader("📊 Prediction Results")
    st.dataframe(df, use_container_width=True)


    st.subheader("📈 Risk Distribution")
    st.bar_chart(df['Predicted_Risk'].value_counts())

    # ================= SHAP EXPLANATION =================
    st.subheader("🔍 SHAP Explanation (Why this prediction?)")

    student_index = st.selectbox("Select Student Index", df.index)

    # ---------- Background ----------
    background = X.sample(min(50, len(X)), random_state=42)

    # ---------- Safe wrapper ----------
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

    # ================= BULLETPROOF SHAP EXTRACTION =================
    # 1. Convert to numpy
    shap_arr = np.array(shap_values, dtype=object)

    # 2. Take absolute values
    shap_arr = np.abs(shap_arr)

    # 3. Flatten EVERYTHING safely
    shap_flat = np.concatenate([
        np.ravel(s) for s in shap_arr
    ])

    # 4. Ensure exact feature length
    if shap_flat.size >= N_FEATURES:
        shap_vector = shap_flat[:N_FEATURES]
    else:
        shap_vector = np.pad(
            shap_flat,
            (0, N_FEATURES - shap_flat.size),
            constant_values=0
        )

    # 5. Create DataFrame (NOW GUARANTEED SAFE)
    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "Impact": shap_vector
    }).sort_values(by="Impact", ascending=False)

    st.write(
        f"🧠 Explanation for predicted class: "
        f"**{df.loc[student_index, 'Predicted_Risk']}**"
    )

    st.dataframe(shap_df, use_container_width=True)

