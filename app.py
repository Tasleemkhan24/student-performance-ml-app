import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# -------------------------------
# Page Config (MUST be first Streamlit call)
# -------------------------------
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(to right, #f8fbff, #ffffff);
    padding: 10px;
}
h1, h2, h3 {
    color: #1E3A8A;
}
.stDownloadButton > button {
    background-color: #1E90FF;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 6px 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & Intro
# -------------------------------
st.title("🎓 Student Performance Prediction")
st.write("Upload student data (CSV) to predict grades using **XGBoost** and **CatBoost** models.")

# -------------------------------
# Load Models & Encoders
# -------------------------------
@st.cache_resource
def load_resources():
    xgb_model = joblib.load("XGBoost_model.pkl")
    cb_model = joblib.load("CatBoost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_y = joblib.load("label_encoder_y.pkl")
    final_features = joblib.load("final_features.pkl")
    numeric_feats = joblib.load("numeric_features.pkl")
    return xgb_model, cb_model, scaler, le_y, final_features, numeric_feats

xgb_model, cb_model, scaler, le_y, final_features, numeric_feats = load_resources()
cat_feats = [c for c in final_features if c not in numeric_feats]

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    st.write("1. Upload your CSV\n2. Preview data\n3. Generate predictions")
    st.markdown("---")
    primary_model = st.selectbox("Show predictions for", ["Both", "XGBoost", "CatBoost"])

# -------------------------------
# Tabs Layout
# -------------------------------
tab_upload, tab_preview, tab_predict, tab_charts = st.tabs(
    ["📂 Upload Data", "🔎 Preview", "🔮 Predictions", "📊 Charts"]
)

# -------------------------------
# Upload Tab
# -------------------------------
with tab_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Please upload a CSV file to continue.")

# -------------------------------
# Preview Tab
# -------------------------------
with tab_preview:
    if uploaded_file:
        st.subheader("Data Preview (First 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("Upload a file first in the 'Upload Data' tab.")

# -------------------------------
# Predictions Tab
# -------------------------------
with tab_predict:
    if uploaded_file:
        with st.spinner("Processing data and generating predictions..."):
            df_pred = df.copy()

            # numeric
            for col in numeric_feats:
                if col not in df_pred.columns:
                    df_pred[col] = 0
                else:
                    df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(0)
            df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

            # categorical
            for col in cat_feats:
                if col not in df_pred.columns:
                    df_pred[col] = 'missing'
                else:
                    df_pred[col] = df_pred[col].fillna('missing').astype(str)
                try:
                    le = joblib.load(f"models/le_{col}.pkl")
                    df_pred[col] = df_pred[col].apply(lambda x: x if x in le.classes_ else 'missing')
                    if 'missing' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'missing')
                    df_pred[col] = le.transform(df_pred[col])
                except:
                    le = LabelEncoder()
                    df_pred[col] = le.fit_transform(df_pred[col])

            df_pred = df_pred[final_features]

            # names
            first_name_col = next((c for c in df_pred.columns if 'first_name' in c.lower()), None)
            last_name_col = next((c for c in df_pred.columns if 'last_name' in c.lower()), None)
            df['Student Name'] = ''
            if first_name_col or last_name_col:
                df['Student Name'] = df.get(first_name_col, '').fillna('') + ' ' + df.get(last_name_col, '').fillna('')
            else:
                df['Student Name'] = 'Unknown'

            # predictions
            preds = {}
            if primary_model in ["Both", "XGBoost"]:
                try:
                    xgb_preds = xgb_model.predict(df_pred)
                    preds['XGBoost_Predicted_Grade'] = le_y.inverse_transform(xgb_preds)
                except Exception as e:
                    st.error(f"XGBoost error: {e}")

            if primary_model in ["Both", "CatBoost"]:
                try:
                    cb_preds = cb_model.predict(df_pred)
                    preds['CatBoost_Predicted_Grade'] = le_y.inverse_transform(cb_preds)
                except Exception as e:
                    st.error(f"CatBoost error: {e}")

            for col, values in preds.items():
                df[col] = values

        # KPI Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Students", f"{df.shape[0]}")
        c2.metric("Missing Values", f"{df.isna().sum().sum()}")
        if "XGBoost_Predicted_Grade" in df.columns:
            c3.metric("Most Common Grade", df["XGBoost_Predicted_Grade"].mode()[0])

        # results
        display_cols = ['Student Name'] + list(preds.keys())
        st.subheader("Prediction Results")
        st.dataframe(df[display_cols], use_container_width=True)

        # download
        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
    else:
        st.warning("Upload a file first to get predictions.")

# -------------------------------
# Charts Tab
# -------------------------------
with tab_charts:
    if uploaded_file:
        if "XGBoost_Predicted_Grade" in df.columns:
            fig = px.histogram(df, x="XGBoost_Predicted_Grade", title="XGBoost Grade Distribution")
            st.plotly_chart(fig, use_container_width=True)

        if "CatBoost_Predicted_Grade" in df.columns:
            fig2 = px.histogram(df, x="CatBoost_Predicted_Grade", title="CatBoost Grade Distribution")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Charts will appear after predictions are made.")
