# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🎓 Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    scaler = None
    le_y = None

    try:
        models["XGBoost"] = joblib.load("XGBoost_model.pkl")
        models["CatBoost"] = joblib.load("CatBoost_model.pkl")
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

    # Optional: try loading scaler/encoder if they exist
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    if os.path.exists("label_encoder_y.pkl"):
        le_y = joblib.load("label_encoder_y.pkl")

    return models, scaler, le_y

models, scaler, le_y = load_models()

if not models:
    st.stop()

# -----------------------------
# Sidebar: Model Choice
# -----------------------------
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

st.markdown("---")

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("📤 Upload Student Dataset for Prediction")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Basic Preprocessing
    # -----------------------------
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())

    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).fillna("missing")
        df[c] = pd.factorize(df[c])[0]

    # Optional scaling
    if scaler:
        try:
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = scaler.transform(df[num_cols])
            st.success("✅ Data scaled successfully using saved scaler.")
        except Exception as e:
            st.warning(f"⚠️ Scaling skipped due to mismatch: {e}")
    else:
        st.info("ℹ️ No scaler found. Using raw numeric values for prediction.")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        try:
            preds = model.predict(df)

            if le_y:
                preds = le_y.inverse_transform(preds)
                df["Predicted_Grade"] = preds
            else:
                df["Predicted_Score"] = preds

            st.write("### 🎯 Prediction Results")
            st.dataframe(df.head())

            # -----------------------------
            # Visualization
            # -----------------------------
            st.subheader("📊 Prediction Summary")

            if "Predicted_Grade" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x="Predicted_Grade", data=df, palette="coolwarm", ax=ax)
                plt.title("Predicted Grade Distribution")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df["Predicted_Score"], kde=True, bins=20, ax=ax)
                plt.title("Predicted Score Distribution")
                st.pyplot(fig)

            # -----------------------------
            # SHAP Explainability
            # -----------------------------
            st.subheader("🧠 SHAP Model Explainability")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df)
                st.write("### 🔍 Feature Importance (SHAP Summary)")
                shap_fig, ax = plt.subplots()
                shap.summary_plot(shap_values, df, plot_type="bar", show=False)
                st.pyplot(shap_fig)
            except Exception as e:
                st.warning(f"⚠️ SHAP visualization skipped: {e}")

            # -----------------------------
            # Download Predictions
            # -----------------------------
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    st.info("👆 Please upload a CSV file to start predictions.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by **Kayamkhani Thasleem** | B.Tech CSE (R20) | 🧠 Streamlit Deployment for Student Performance ML Models")
