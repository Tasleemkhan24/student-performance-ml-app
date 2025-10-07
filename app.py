import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
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
    # Fill missing values
    # -----------------------------
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].fillna("missing")

    # -----------------------------
    # Encode categorical features
    # -----------------------------
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        st.info(f"🔡 Encoding categorical columns: {list(cat_cols)}")
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        st.info("ℹ️ No categorical columns detected.")

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("🔮 Predict Performance"):
        try:
            # Check missing columns
            missing_cols = [col for col in expected_features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns required for prediction: {missing_cols}")
                st.stop()

            # Select features in correct order
            X_pred = df[expected_features]

            # Perform prediction
            preds = model.predict(X_pred)

            # Combine first and last name if available
            name_cols = [col for col in df.columns if 'First_Name' in col or 'Last_Name' in col]
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
