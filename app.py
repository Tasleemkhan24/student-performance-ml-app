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
st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['XGBoost'] = joblib.load("models/best_xgb.pkl")
        models['CatBoost'] = joblib.load("models/best_catboost.pkl")
        models['AdaBoost'] = joblib.load("models/best_adaboost.pkl")
        scaler = joblib.load("models/scaler.pkl")
        le_y = joblib.load("models/label_encoder_y.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        models, scaler, le_y = {}, None, None
    return models, scaler, le_y

models, scaler, le_y = load_models()

if not models:
    st.stop()

model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))
model = models[model_choice]

st.markdown("---")

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("📤 Upload Student Dataset for Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # -----------------------------
    # Preprocessing (light)
    # -----------------------------
    # Handle missing numeric values
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())

    # Encode categoricals if present
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).fillna("missing")
        try:
            df[c] = pd.factorize(df[c])[0]
        except:
            pass

    # Scale numeric features
    num_feats = df.select_dtypes(include=np.number).columns.tolist()
    df[num_feats] = scaler.transform(df[num_feats])

    st.success("✅ Preprocessing completed successfully!")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        preds = model.predict(df)

        # Decode target if classification
        try:
            preds_decoded = le_y.inverse_transform(preds)
            df['Predicted_Grade'] = preds_decoded
        except Exception:
            df['Predicted_Score'] = preds

        st.write("### 🎯 Prediction Results", df.head())

        # -----------------------------
        # Visualization
        # -----------------------------
        st.subheader("📊 Prediction Summary")

        if 'Predicted_Grade' in df.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(x='Predicted_Grade', data=df, palette='coolwarm', ax=ax)
            plt.title("Predicted Grade Distribution")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df['Predicted_Score'], kde=True, bins=20, ax=ax)
            plt.title("Predicted Score Distribution")
            st.pyplot(fig)

        # -----------------------------
        # SHAP Explainability
        # -----------------------------
        st.subheader("🧠 Model Explainability (SHAP)")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df)

            st.write("Feature Importance (SHAP Summary)")
            shap_fig, ax = plt.subplots()
            shap.summary_plot(shap_values, df, plot_type="bar", show=False)
            st.pyplot(shap_fig)

        except Exception as e:
            st.warning(f"SHAP visualization skipped: {e}")

        # Option to download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to start predictions.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Kayamkhani Thasleem | B.Tech CSE (R20) | 🎓 Streamlit Deployment for Student Performance ML Pipeline")
