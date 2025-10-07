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
# Dark Theme + Animations
# -------------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: #ffffff; }
[data-testid="stSidebar"] { background: rgba(20,20,20,0.9); backdrop-filter: blur(6px); border-right: 2px solid #0f2027; }

.top-banner { background-color: rgba(30,144,255,0.9); padding:20px; border-radius:12px; box-shadow:0 4px 15px rgba(0,0,0,0.3); text-align:center; margin-bottom:25px; transition: transform 0.3s;}
.top-banner:hover { transform: scale(1.02); }

h1,h2,h3,h4 { color:#66ccff; font-family:'Segoe UI',sans-serif; font-weight:700; }

.stButton>button { background-color:#1e90ff; color:white; border-radius:10px; height:3em; width:100%; font-size:16px; font-weight:bold; box-shadow:0 0 10px #1e90ff; transition: all 0.3s ease; }
.stButton>button:hover { background-color:#005f99; box-shadow:0 0 20px #00bfff; transform: scale(1.05); }

.stFileUploader>div>div { color:#ffffff; }

.badge { display:inline-block; padding:4px 12px; border-radius:12px; font-weight:bold; margin:2px; cursor:pointer; transition: all 0.3s ease; }
.badge:hover { transform: scale(1.2); box-shadow: 0 0 10px #ffffff; }
.badge-A { background-color:#00c853; color:#ffffff; }
.badge-B { background-color:#76ff03; color:#000000; }
.badge-C { background-color:#ffeb3b; color:#000000; }
.badge-D { background-color:#ff9800; color:#ffffff; }
.badge-F { background-color:#d50000; color:#ffffff; }

.kpi-card { padding:20px; border-radius:12px; background-color: rgba(30,30,30,0.7); box-shadow: 0 4px 15px rgba(0,0,0,0.3); transition: transform 0.3s ease, box-shadow 0.3s ease; text-align:center; }
.kpi-card:hover { transform: scale(1.05); box-shadow: 0 0 20px #1e90ff; }
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
# Load Models
# -------------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")
final_features = joblib.load("final_features.pkl")
numeric_feats = joblib.load("numeric_features.pkl")
cat_feats = [c for c in final_features if c not in numeric_feats]

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    primary_model = st.selectbox("Show predictions for", ["Both", "XGBoost", "CatBoost"])
    st.markdown("---")

# -------------------------------
# Session State
# -------------------------------
if "filter_grade" not in st.session_state:
    st.session_state.filter_grade = "All"

# -------------------------------
# Tabs
# -------------------------------
tab_upload, tab_preview, tab_predictions, tab_charts = st.tabs(
    ["📂 Upload Data","🔎 Preview","🔮 Predictions","📊 Charts"]
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
    if uploaded_file: st.dataframe(df.head(50), use_container_width=True)
    else: st.warning("Upload a file first.")

# -------------------------------
# Predictions Tab
# -------------------------------
with tab_predictions:
    if uploaded_file:
        with st.spinner("Generating predictions..."):
            df_pred = df.copy()
            for col in numeric_feats:
                df_pred[col] = pd.to_numeric(df_pred.get(col,0), errors='coerce').fillna(0)
            df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

            for col in cat_feats:
                df_pred[col] = df_pred.get(col,'missing').fillna('missing').astype(str)
                try:
                    le = joblib.load(f"models/le_{col}.pkl")
                    df_pred[col] = df_pred[col].apply(lambda x: x if x in le.classes_ else 'missing')
                    if 'missing' not in le.classes_: le.classes_ = np.append(le.classes_,'missing')
                    df_pred[col] = le.transform(df_pred[col])
                except:
                    le = LabelEncoder(); df_pred[col] = le.fit_transform(df_pred[col])
            df_pred = df_pred[final_features]

            first_name_col = next((c for c in df_pred.columns if 'first_name' in c.lower()), None)
            last_name_col = next((c for c in df_pred.columns if 'last_name' in c.lower()), None)
            df['Student Name'] = df.get(first_name_col,'').fillna('') + ' ' + df.get(last_name_col,'').fillna('') if first_name_col or last_name_col else 'Unknown'

            preds = {}
            if primary_model in ["Both","XGBoost"]:
                xgb_preds = xgb_model.predict(df_pred)
                preds['XGBoost'] = le_y.inverse_transform(xgb_preds)
            if primary_model in ["Both","CatBoost"]:
                cb_preds = cb_model.predict(df_pred)
                preds['CatBoost'] = le_y.inverse_transform(cb_preds)
            for col, val in preds.items(): df[col] = val

        # KPI Cards with animation
        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='kpi-card'><h3>Total Students</h3><h2>{df.shape[0]}</h2></div>",unsafe_allow_html=True)
        c2.markdown(f"<div class='kpi-card'><h3>Missing Values</h3><h2>{df.isna().sum().sum()}</h2></div>",unsafe_allow_html=True)
        if "XGBoost" in df.columns: c3.markdown(f"<div class='kpi-card'><h3>Top Grade</h3><h2>{df['XGBoost'].mode()[0]}</h2></div>",unsafe_allow_html=True)

        # Animated badges
        def grade_badge(val, tooltip):
            badge_class = f"badge-{val}" if val in ['A','B','C','D','F'] else ""
            return f"<span class='badge {badge_class}' title='{tooltip}'>{val}</span>"

        display_df = df.copy()
        display_df['XGBoost_Badge'] = display_df['XGBoost'].apply(lambda x: grade_badge(x,f"XGBoost: {x}") if 'XGBoost' in display_df else '')
        display_df['CatBoost_Badge'] = display_df['CatBoost'].apply(lambda x: grade_badge(x,f"CatBoost: {x}") if 'CatBoost' in display_df else '')
        display_df['Predictions'] = display_df[['XGBoost_Badge','CatBoost_Badge']].agg(' '.join, axis=1)

        st.subheader("Prediction Results")
        st.write(display_df[['Student Name','Predictions']].to_html(escape=False,index=False), unsafe_allow_html=True)

        # Download CSV
        csv = display_df[['Student Name']+list(preds.keys())].to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

# -------------------------------
# Charts Tab
# -------------------------------
with tab_charts:
    if uploaded_file and preds:
        st.subheader("Grade Distribution Charts")
        filtered_df = df.copy()
        if st.session_state.filter_grade != "All":
            filtered_df = filtered_df[filtered_df[list(preds.keys())[0]]==st.session_state.filter_grade]

        for model in preds.keys():
            color = ['#00c853','#76ff03','#ffeb3b','#ff9800','#d50000']
            fig = px.histogram(filtered_df, x=model, title=f"{model} Distribution", color=model, color_discrete_sequence=color)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Charts will appear after predictions.")
