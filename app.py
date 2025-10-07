import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================
# 🎯 App Title
# =========================================
st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# =========================================
# ⚙️ Load Models from GitHub Repo (local path)
# =========================================
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

# =========================================
# 📤 Upload CSV File
# =========================================
st.sidebar.header("📂 Upload Student Dataset for Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # =========================================
    # 🧠 Select Model
    # =========================================
    model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "CatBoost"])
    model = models.get(model_choice)

    # =========================================
    # 🔍 Preprocessing (Remove Name/Email columns)
    # =========================================
    excluded_columns = ['First_Name', 'Last_Name', 'Email']
    input_df = df.drop(columns=[col for col in excluded_columns if col in df.columns], errors='ignore')

    # Identify categorical columns
    cat_cols = input_df.select_dtypes(include=['object']).columns

    # One-hot encode categorical variables
    if len(cat_cols) > 0:
        input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

    # =========================================
    # 🚀 Prediction
    # =========================================
    try:
        y_pred = model.predict(input_df)

        # Add predictions back to original data (with names)
        result_df = pd.DataFrame()
        if 'First_Name' in df.columns and 'Last_Name' in df.columns:
            result_df['Student_Name'] = df['First_Name'].astype(str) + " " + df['Last_Name'].astype(str)
        else:
            result_df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]

        result_df['Prediction'] = y_pred

        st.subheader("🎯 Prediction Results")
        st.dataframe(result_df)

        # Download Results
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", data=csv, file_name="predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.info("📥 Please upload a CSV file to begin prediction.")
