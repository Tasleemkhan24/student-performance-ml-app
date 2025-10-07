import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(page_title="🎓 Student Performance Predictor", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #f6fbff 0%, #ffffff 100%); padding-top:10px; }
h1 { color: #0f1724; text-align:center; }
.stDownloadButton>button { background-color: #1E90FF; color: white; border-radius: 8px; padding: 8px 14px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Predictor")
st.write("Upload a CSV with student features — predictions from XGBoost & CatBoost will be shown.")

# ----------------- load models (cached) -----------------
@st.cache_resource
def load_resources():
    xgb = joblib.load("XGBoost_model.pkl")
    cb = joblib.load("CatBoost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_y = joblib.load("label_encoder_y.pkl")
    final_features = joblib.load("final_features.pkl")
    numeric_feats = joblib.load("numeric_features.pkl")
    return xgb, cb, scaler, le_y, final_features, numeric_feats

xgb_model, cb_model, scaler, le_y, final_features, numeric_feats = load_resources()
cat_feats = [c for c in final_features if c not in numeric_feats]

# Sidebar
with st.sidebar:
    st.header("📌 Instructions")
    st.write("Upload a CSV that contains the features required by the model.")
    st.markdown("---")
    primary_model = st.selectbox("Show predictions for", ["Both", "XGBoost", "CatBoost"])

# Tabs
tab_u, tab_preview, tab_preds, tab_charts = st.tabs(["📁 Upload", "🔎 Preview", "🔮 Predictions", "📊 Charts"])

df = pd.DataFrame()
with tab_u:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)

with tab_preview:
    if df.empty:
        st.info("Upload a CSV in the Upload tab to preview it here.")
    else:
        st.subheader("Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

with tab_preds:
    if df.empty:
        st.info("Upload data first.")
    else:
        with st.spinner("Preprocessing and predicting..."):
            # copy and fill numeric features
            df_pred = df.copy()
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
            try:
                xgb_preds = xgb_model.predict(df_pred)
                preds['XGBoost_Predicted_Grade'] = le_y.inverse_transform(xgb_preds)
            except Exception as e:
                st.error(f"XGBoost error: {e}")
            try:
                cb_preds = cb_model.predict(df_pred)
                preds['CatBoost_Predicted_Grade'] = le_y.inverse_transform(cb_preds)
            except Exception as e:
                st.error(f"CatBoost error: {e}")

            for k, v in preds.items():
                df[k] = v

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]}")
        c2.metric("Missing values", f"{df.isna().sum().sum()}")
        if 'XGBoost_Predicted_Grade' in df.columns:
            c3.metric("Top grade (XGB)", str(df['XGBoost_Predicted_Grade'].mode()[0]))

        # show results (interactive table could be used instead)
        display_cols = ['Student Name'] + list(preds.keys())
        st.subheader("Predictions")
        st.dataframe(df[display_cols], use_container_width=True)

        csv = df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv, file_name="predictions.csv", mime="text/csv")

with tab_charts:
    if "XGBoost_Predicted_Grade" in df.columns:
        fig = px.histogram(df, x="XGBoost_Predicted_Grade", title="XGBoost Grade Distribution")
        st.plotly_chart(fig, use_container_width=True)
    if "CatBoost_Predicted_Grade" in df.columns:
        fig2 = px.histogram(df, x="CatBoost_Predicted_Grade", title="CatBoost Grade Distribution")
        st.plotly_chart(fig2, use_container_width=True)
