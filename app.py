import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

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

st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    st.warning("⚠️ Feature names not found in the model. Make sure CSV columns match training features.")
    expected_features = []

st.markdown("---")

st.subheader("📤 Upload Student Dataset for Prediction")
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # Handle missing values
    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].fillna("missing")

    # Exclude name/email columns from encoding
    name_related_cols = [c for c in df.columns if any(x in c.lower() for x in ['name', 'email'])]
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c not in name_related_cols]

    # Encode only non-name categorical columns
    if len(cat_cols) > 0:
        st.info(f"🔡 Encoding categorical columns (excluding names/emails): {cat_cols}")
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        st.info("ℹ️ No categorical columns detected for encoding.")

    if st.button("🔮 Predict Performance"):
        try:
            missing_cols = [col for col in expected_features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns required for prediction: {missing_cols}")
                st.stop()

            X_pred = df[expected_features]

            preds = model.predict(X_pred)

            # Combine first and last name for readable display
            first_name_col = [c for c in df.columns if 'first_name' in c.lower()]
            last_name_col = [c for c in df.columns if 'last_name' in c.lower()]
            if first_name_col or last_name_col:
                df['Student_Name'] = df.get(first_name_col[0], "") + " " + df.get(last_name_col[0], "")
            else:
                df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]

            results = pd.DataFrame({
                "Student_Name": df['Student_Name'],
                "Prediction": preds
            })

            st.success("✅ Prediction completed successfully!")
            st.write("### 🎯 Prediction Results (Student Name + Prediction)", results.head())

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

            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv, "student_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
else:
    st.info("👆 Upload a CSV file to start predictions.")

st.markdown("---")
st.caption("Developed by Kayamkhani Thasleem | B.Tech CSE (R20) | 🧠 Streamlit Deployment for Student Performance ML Models")
