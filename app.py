import streamlit as st
import pandas as pd
import joblib
import shap
import os
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Student Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Student Performance Risk Dashboard")
st.caption(
    "An easy-to-use system to identify academically at-risk students "
    "and understand *why* the prediction was made."
)

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
st.sidebar.header("🧭 How to use")
st.sidebar.markdown(
    """
    **Step 1:** Upload student data  
    **Step 2:** View risk predictions  
    **Step 3:** Explore why a student is at risk  
    """
)

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Student Data (CSV / Excel)",
    type=["csv", "xlsx"]
)

# ================= MAIN APP =================
if uploaded_file is None:
    st.info("⬅️ Start by uploading a student data file from the sidebar.")
    st.stop()

# ---------- Load data ----------
df = (
    pd.read_excel(uploaded_file)
    if uploaded_file.name.endswith(".xlsx")
    else pd.read_csv(uploaded_file)
)

# ---------- Feature engineering ----------
quiz_cols = ['quiz_1','quiz_2','quiz_3','quiz_4','quiz_5']
df['quiz_avg'] = df[quiz_cols].mean(axis=1)
df['quiz_std'] = df[quiz_cols].std(axis=1)

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# ---------- Prediction ----------
preds = rf_model.predict(X)
probs = rf_model.predict_proba(X)

df['Predicted_Risk'] = label_encoder.inverse_transform(preds)

for i, cls in enumerate(label_encoder.classes_):
    df[f'Prob_{cls}'] = probs[:, i]

# ================= SUMMARY METRICS =================
st.divider()
st.subheader("📌 Quick Overview")

col1, col2, col3 = st.columns(3)
col1.metric("👨‍🎓 Total Students", len(df))
col2.metric("⚠️ At Risk", (df["Predicted_Risk"] == "AtRisk").sum())
col3.metric("🚨 Critical", (df["Predicted_Risk"] == "Critical").sum())

# ================= FILTER & SEARCH =================
st.divider()
st.subheader("🔍 Explore Students")

risk_filter = st.selectbox(
    "Filter by Risk Level",
    ["All"] + sorted(df["Predicted_Risk"].unique().tolist())
)

if risk_filter != "All":
    df_view = df[df["Predicted_Risk"] == risk_filter]
else:
    df_view = df.copy()

# ================= RESULTS TABLE =================
with st.expander("📊 Student Prediction Results", expanded=True):
    st.dataframe(df_view, use_container_width=True)

    st.download_button(
        "⬇️ Download Results",
        df_view.to_csv(index=False),
        "student_risk_predictions.csv",
        "text/csv"
    )

# ================= RISK DISTRIBUTION =================
st.divider()
st.subheader("📈 Risk Distribution (Class View)")

st.bar_chart(
    df["Predicted_Risk"].value_counts(),
    use_container_width=True
)

# ================= SHAP EXPLANATION =================
st.divider()
st.subheader("🧠 Why is this student at risk?")

student_index = st.selectbox(
    "Select a student (by row index)",
    df.index
)

st.success(
    f"Predicted Risk Level: **{df.loc[student_index, 'Predicted_Risk']}**"
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
        "Impact Score": shap_vector
    })
    .sort_values(by="Impact Score", ascending=False)
    .head(8)
)

with st.expander("📌 Key Factors Influencing This Prediction", expanded=True):
    st.dataframe(shap_df, use_container_width=True)

st.caption(
    "Higher impact score = stronger influence on the model’s decision"
)

# ================= FOOTER =================
st.divider()
st.caption(
    "🎯 This tool is designed to support educators and institutions "
    "in early identification and intervention."
)
