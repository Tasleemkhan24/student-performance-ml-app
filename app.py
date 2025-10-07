# app_final_fixed_names.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction Dashboard")

# -------------------------------------------------------
# Load models and artifacts safely from 'models/' folder
# -------------------------------------------------------
@st.cache_resource
def load_artifact(path):
    """Helper to load artifact if it exists"""
    try:
        if os.path.exists(path):
            return joblib.load(path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None


st.sidebar.header("⚙️ Model & Artifact Settings")
model_path = st.sidebar.text_input("Model Path", value="models/XGBoost_model.pkl")
scaler_path = st.sidebar.text_input("Scaler Path (optional)", value="models/scaler.pkl")
encoders_path = st.sidebar.text_input("Encoders Path (optional)", value="models/encoders.pkl")
features_path = st.sidebar.text_input("Feature List Path (optional)", value="models/feature_list.pkl")
target_le_path = st.sidebar.text_input("Target Encoder Path (optional)", value="models/label_encoder_y.pkl")

model = load_artifact(model_path)
scaler = load_artifact(scaler_path)
encoders = load_artifact(encoders_path)
feature_list = load_artifact(features_path)
le_target = load_artifact(target_le_path)

if model is None:
    st.error(f"❌ Model not found at {model_path}. Please upload or set correct path.")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# -------------------------------------------------------
# Upload Prediction CSV
# -------------------------------------------------------
st.markdown("---")
st.subheader("📤 Upload Student Dataset for Prediction")
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=['csv'])

if uploaded_file is None:
    st.info("👆 Upload a CSV file to start predictions.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.write("### Uploaded Data (Preview)", df.head())

# -------------------------------------------------------
# Separate name/email columns (for display only)
# -------------------------------------------------------
name_cols = [c for c in df.columns if any(x in c.lower() for x in ['first_name', 'last_name', 'email'])]
df_names = df[name_cols].astype(str).fillna('') if len(name_cols) > 0 else None

# Drop name/email columns before preprocessing
df_features = df.drop(columns=name_cols, errors='ignore')

# -------------------------------------------------------
# Handle missing values
# -------------------------------------------------------
for c in df_features.select_dtypes(include=np.number).columns:
    df_features[c] = df_features[c].fillna(df_features[c].mean())
for c in df_features.select_dtypes(include='object').columns:
    df_features[c] = df_features[c].fillna('missing')

# -------------------------------------------------------
# If encoders missing, build them from uploaded CSV
# -------------------------------------------------------
if encoders is None:
    st.warning("⚠️ Encoders not found. Building from uploaded CSV (may affect accuracy).")
    encoders = {}
    cat_cols = df_features.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        le = LabelEncoder()
        df_features[c] = df_features[c].astype(str)
        le.fit(df_features[c])
        encoders[c] = le
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders_built_from_pred_csv.pkl")
    st.info("Encoders built and saved to models/encoders_built_from_pred_csv.pkl")

# -------------------------------------------------------
# Determine final feature list
# -------------------------------------------------------
if feature_list is None:
    if hasattr(model, "feature_names_in_"):
        feature_list = list(model.feature_names_in_)
        st.info("✅ Using model.feature_names_in_ as feature order.")
    else:
        num_feats = df_features.select_dtypes(include=np.number).columns.tolist()
        cat_feats = [c for c in df_features.select_dtypes(include='object').columns.tolist()]
        feature_list = num_feats + cat_feats
        st.warning("⚠️ Feature list inferred automatically (may not match training).")
    joblib.dump(feature_list, "models/feature_list_inferred.pkl")

st.write("### Expected features:", len(feature_list))

# -------------------------------------------------------
# Prepare input features in correct order
# -------------------------------------------------------
X_work = pd.DataFrame(index=df_features.index)

for feat in feature_list:
    if feat not in df_features.columns:
        st.warning(f"Feature '{feat}' missing in uploaded CSV → filled with 0")
        X_work[feat] = 0
        continue

    # Encode categorical
    if feat in encoders:
        le = encoders[feat]
        vals = df_features[feat].astype(str).tolist()
        transformed = []
        for v in vals:
            if v in le.classes_:
                transformed.append(int(le.transform([v])[0]))
            else:
                transformed.append(0)  # unseen value fallback
        X_work[feat] = transformed
    else:
        # numeric
        try:
            X_work[feat] = pd.to_numeric(df_features[feat])
        except Exception:
            X_work[feat] = 0

X_pred = X_work.reindex(columns=feature_list, fill_value=0)
st.write("### Processed Features (Preview)", X_pred.head())

# -------------------------------------------------------
# Apply Scaler (if available)
# -------------------------------------------------------
if scaler is not None:
    try:
        numeric_cols = X_pred.select_dtypes(include=[np.number]).columns.tolist()
        X_scaled = X_pred.copy()
        X_scaled[numeric_cols] = scaler.transform(X_pred[numeric_cols])
        X_input = X_scaled
        st.info("✅ Scaler applied to numeric features.")
    except Exception as e:
        st.error(f"Scaler application failed: {e}. Proceeding without scaling.")
        X_input = X_pred
else:
    X_input = X_pred

st.write("### Final Input to Model", X_input.head())

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
if st.button("🔮 Predict Performance"):
    try:
        preds = model.predict(X_input)

        if le_target is not None:
            try:
                preds_display = le_target.inverse_transform(preds)
            except Exception:
                preds_display = preds
        else:
            preds_display = preds

        # Create readable student names
        if df_names is not None:
            if {'first_name_x', 'last_name_x'} <= set(df_names.columns.str.lower()):
                df['Student_Name'] = df_names['First_Name_x'] + ' ' + df_names['Last_Name_x']
            elif {'first_name', 'last_name'} <= set(df_names.columns.str.lower()):
                df['Student_Name'] = df_names['first_name'] + ' ' + df_names['last_name']
            else:
                df['Student_Name'] = df_names.astype(str).agg(' '.join, axis=1)
        else:
            df['Student_Name'] = [f"Student_{i+1}" for i in range(len(df))]

        # Combine results
        results = pd.DataFrame({
            "Student_Name": df['Student_Name'],
            "Prediction": preds_display
        })

        st.success("✅ Prediction Completed Successfully!")
        st.write("### 🎯 Prediction Results", results.head(60))

        # Visualization
        st.subheader("📊 Prediction Summary")
        st.bar_chart(results['Prediction'].value_counts())

        # Download
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "student_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Developed by Kayamkhani Thasleem | B.Tech CSE (R20) | Streamlit Cloud Deployment for Student ML Models")
