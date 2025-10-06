# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
        models['XGBoost'] = joblib.load("XGBoost_model.pkl")
        models['CatBoost'] = joblib.load("CatBoost_model.pkl")
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
    return models

models = load_models()

if not models:
    st.stop()

# -----------------------------
# Model Selection
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
    st.write("### Uploaded Data Preview", df.head())

    # -----------------------------
    # Basic Preprocessing
    # -----------------------------
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())

    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].fillna("missing")

    st.info("ℹ️ Using raw input data — no scaling or encoding applied.")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        try:
            preds = model.predict(df)

            # Determine which column might contain student names
            name_col = None
            for col in df.columns:
                if 'name' in col.lower() or 'student' in col.lower():
                    name_col = col
                    break

            if name_col is None:
                st.warning("⚠️ No 'Name' column found — adding a generic ID column instead.")
                df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]
                name_col = 'Student_Name'

            # Create a clean results DataFrame
            results = pd.DataFrame({
                "Student_Name": df[name_col],
                "Predicted_Output": preds
            })

            st.success("✅ Prediction completed successfully!")
            st.write("### 🎯 Prediction Results (Name + Prediction)", results.head())

            # -----------------------------
            # Visualization
            # -----------------------------
            st.subheader("📊 Prediction Summary")

            if results['Predicted_Output'].dtype == 'object' or len(results['Predicted_Output'].unique()) < 10:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Predicted_Output', data=results, palette='coolwarm', ax=ax)
                plt.title("Predicted Category Distribution")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(results['Predicted_Output'], kde=True, bins=20, ax=ax)
                plt.title("Predicted Value Distribution")
                st.pyplot(fig)

            # -----------------------------
            # Download Results
            # -----------------------------
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "student_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    st.info("👆 Upload a CSV file to start predictions.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed by Kayamkhani Thasleem | B.Tech CSE (R20) | 🧠 Streamlit Deployment for Student Performance ML Models")
