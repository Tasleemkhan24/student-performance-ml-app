import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="🎓 Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Dark Theme + Banner Styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background: rgba(20, 20, 20, 0.9);
    backdrop-filter: blur(6px);
    border-right: 2px solid #0f2027;
}
.top-banner {
    background-color: rgba(30, 144, 255, 0.9);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    text-align: center;
    margin-bottom: 25px;
}
.top-banner h1 { color: #ffffff; margin: 0; font-size: 36px; }
.top-banner p { color: #e0f7ff; font-size: 18px; margin: 5px 0 0 0; }
h1, h2, h3, h4 { color: #66ccff; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
.stButton > button {
    background-color: #1e90ff; color: white; border-radius: 10px; height: 3em; width: 100%;
    font-size: 16px; font-weight: bold; box-shadow: 0 0 10px #1e90ff;
}
.stButton > button:hover { background-color: #005f99; box-shadow: 0 0 15px #00bfff; }
.stFileUploader>div>div { color: #ffffff; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-weight: bold;
    margin: 2px;
}
.badge-xgb { background-color: #1e90ff; color: #ffffff; }
.badge-cb { background-color: #66ccff; color: #000000; }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Top Banner
# -------------------------------
st.markdown("""
<div class="top-banner">
    <h1>🎓 Student Performance Predictor</h1>
    <p>Upload student data and predict grades with XGBoost & CatBoost</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Models & Encoders
# -------------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")
final_features = joblib.load("final_features.pkl")
numeric_feats = joblib.load("numeric_features.pkl")
cat_feats = [c for c in final_features if c not in numeric_feats]

# -------------------------------
# Sidebar Options
# -------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    primary_model = st.selectbox("Show predictions for", ["Both", "XGBoost", "CatBoost"])
    st.markdown("---")

# -------------------------------
# Tabs
# -------------------------------
tab_upload, tab_preview, tab_predictions, tab_charts = st.tabs(
    ["📂 Upload Data", "🔎 Preview", "🔮 Predictions", "📊 Charts"]
)

# -------------------------------
# Upload Tab
# -------------------------------
with tab_upload:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Uploaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
    else:
        st.info("Please upload a CSV file to start predictions.")

# -------------------------------
# Preview Tab
# -------------------------------
with tab_preview:
    if uploaded_file:
        st.subheader("Data Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("Upload a file first in 'Upload Data' tab.")

# -------------------------------
# Predictions Tab
# -------------------------------
with tab_predictions:
    if uploaded_file:
        with st.spinner("Processing predictions..."):
            df_pred = df.copy()
            # Numeric preprocessing
            for col in numeric_feats:
                if col not in df_pred.columns: df_pred[col] = 0
                else: df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(0)
            df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

            # Categorical preprocessing
            for col in cat_feats:
                if col not in df_pred.columns: df_pred[col] = 'missing'
                else: df_pred[col] = df_pred[col].fillna('missing').astype(str)
                try:
                    le = joblib.load(f"models/le_{col}.pkl")
                    df_pred[col] = df_pred[col].apply(lambda x: x if x in le.classes_ else 'missing')
                    if 'missing' not in le.classes_: le.classes_ = np.append(le.classes_, 'missing')
                    df_pred[col] = le.transform(df_pred[col])
                except:
                    le = LabelEncoder(); df_pred[col] = le.fit_transform(df_pred[col])
            df_pred = df_pred[final_features]

            # Combine names
            first_name_col = next((c for c in df_pred.columns if 'first_name' in c.lower()), None)
            last_name_col = next((c for c in df_pred.columns if 'last_name' in c.lower()), None)
            df['Student Name'] = df.get(first_name_col, '').fillna('') + ' ' + df.get(last_name_col, '').fillna('') if first_name_col or last_name_col else 'Unknown'

            # Generate predictions
            preds = {}
            if primary_model in ["Both", "XGBoost"]:
                xgb_preds = xgb_model.predict(df_pred)
                preds['XGBoost_Predicted_Grade'] = le_y.inverse_transform(xgb_preds)
            if primary_model in ["Both", "CatBoost"]:
                cb_preds = cb_model.predict(df_pred)
                preds['CatBoost_Predicted_Grade'] = le_y.inverse_transform(cb_preds)
            for col, values in preds.items(): df[col] = values

        # KPI Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Students", f"{df.shape[0]}")
        c2.metric("Missing Values", f"{df.isna().sum().sum()}")
        if "XGBoost_Predicted_Grade" in df.columns: c3.metric("Top Grade", df["XGBoost_Predicted_Grade"].mode()[0])

        # Filter by grade
        if preds:
            all_grades = sorted(set(df[list(preds.keys())[0]]))
            selected_grade = st.selectbox("Filter students by grade", ["All"] + all_grades)
            display_df = df.copy()
            if selected_grade != "All":
                display_df = display_df[display_df[list(preds.keys())[0]] == selected_grade]

            # Colored badges for predictions
            def make_badges(row):
                badges = []
                if 'XGBoost_Predicted_Grade' in row: badges.append(f"<span class='badge badge-xgb'>{row['XGBoost_Predicted_Grade']}</span>")
                if 'CatBoost_Predicted_Grade' in row: badges.append(f"<span class='badge badge-cb'>{row['CatBoost_Predicted_Grade']}</span>")
                return " ".join(badges)

            display_df['Predictions'] = display_df.apply(make_badges, axis=1)
            st.subheader("Prediction Results")
            st.write(display_df[['Student Name', 'Predictions']].to_html(escape=False, index=False), unsafe_allow_html=True)

            # Download CSV
            csv = display_df[['Student Name'] + list(preds.keys())].to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    else:
        st.warning("Upload a file first to get predictions.")

# -------------------------------
# Charts Tab
# -------------------------------
with tab_charts:
    if uploaded_file and preds:
        st.subheader("Grade Distribution Charts")
        for model in preds.keys():
            fig = px.histogram(df, x=model, title=f"{model} Distribution", color_discrete_sequence=['#1e90ff' if 'XGBoost' in model else '#66ccff'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Charts will appear after predictions are made.")
