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

# -----------------------------
# Expected Features
# -----------------------------
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    st.warning("⚠️ Feature names not found in the model. Make sure CSV columns match training features.")
    expected_features = []

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
    # Auto Column Cleaning
    # -----------------------------
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
    df.columns = [col.replace('_x', '').replace('_y', '') for col in df.columns]  # clean _x/_y

    # Fill missing numeric and object values
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].fillna("missing")

    st.info("ℹ️ Using raw numeric input features — categorical columns are ignored for prediction.")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        try:
            # Select only numeric features present in model
            numeric_features = [f for f in expected_features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
            if not numeric_features:
                st.error("❌ No numeric columns found for prediction matching model features.")
                st.stop()

            X_pred = df[numeric_features]
            preds = model.predict(X_pred)

            # Combine first and last name if available
            name_cols = [col for col in df.columns if 'first_name' in col.lower() or 'last_name' in col.lower()]
            if name_cols:
                df['Student_Name'] = df[name_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            else:
                df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]

            # Prepare results
            results = pd.DataFrame({
                "Student_Name": df['Student_Name'],
                "Prediction": preds
            })

            st.success("✅ Prediction completed successfully!")
            st.write("### 🎯 Prediction Results (Student Name + Prediction)", results.head())

            # -----------------------------
            # Visualization
            # -----------------------------
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
