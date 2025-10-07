import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Sidebar model selection
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

# Get feature list
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    expected_features = []
    st.warning("⚠️ Model feature names not found. Make sure CSV columns match training features.")

st.markdown("---")

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("📤 Upload Student Dataset for Prediction")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Fill missing values
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].fillna("missing")

    # Identify name/email columns
    name_related_cols = [c for c in df.columns if any(x in c.lower() for x in ['name', 'email'])]

    # Keep copy for displaying names later
    df_names = df[name_related_cols].copy() if name_related_cols else None

    st.info("ℹ️ Dropping non-numeric columns before prediction (like names/emails).")

    # Select numeric columns only for prediction
    df_numeric = df.select_dtypes(include=[np.number])

    # Match expected features (if available)
    if expected_features:
        missing_cols = [col for col in expected_features if col not in df_numeric.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing numeric columns expected by the model: {missing_cols}")
        X_pred = df_numeric.reindex(columns=expected_features, fill_value=0)
    else:
        X_pred = df_numeric

    # -----------------------------
    # Predict
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        try:
            preds = model.predict(X_pred)

            # Combine name columns for display
            if df_names is not None:
                df['Student_Name'] = df_names.apply(lambda x: ' '.join(x.astype(str)), axis=1)
            else:
                df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]

            results = pd.DataFrame({
                "Student_Name": df['Student_Name'],
                "Prediction": preds
            })

            st.success("✅ Prediction completed successfully!")
            st.write("### 🎯 Prediction Results (Student Name + Prediction)", results.head(60))

            # Visualization
            st.subheader("📊 Prediction Summary")
            if results['Prediction'].dtype == 'object' or len(results['Prediction'].unique()) < 10:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Prediction', data=results, palette='coolwarm', ax=ax)
                plt.title("Predicted Category Distribution")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(results['Prediction'], kde=True, bins=20, ax=ax)
                plt.title("Predicted Value Distribution")
                st.pyplot(fig)

            # Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "student_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

else:
    st.info("👆 Upload a CSV file to start predictions.")

st.markdown("---")
st.caption("Developed by Kayamkhani Thasleem | B.Tech CSE (R20) | 🧠 Streamlit Deployment for Student Performance ML Models")
