import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Load Models & Encoders
# -------------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")  # for target inverse transform

# Load feature lists
final_features = joblib.load("final_features.pkl")
numeric_feats = joblib.load("numeric_features.pkl")
cat_feats = [c for c in final_features if c not in numeric_feats]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Student Performance Prediction")
st.write("Upload a CSV file with student data to predict grades.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(10))

    df_pred = df.copy()

    # ---------------------------
    # Preprocess numeric columns
    # ---------------------------
    for col in numeric_feats:
        if col not in df_pred.columns:
            df_pred[col] = 0
        else:
            df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(0)

    # Scale numeric features
    df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

    # ---------------------------
    # Preprocess categorical columns
    # ---------------------------
    for col in cat_feats:
        if col not in df_pred.columns:
            df_pred[col] = 'missing'
        else:
            df_pred[col] = df_pred[col].fillna('missing')
        df_pred[col] = df_pred[col].astype(str)

        # Load per-column LabelEncoder if exists
        try:
            le = joblib.load(f"models/le_{col}.pkl")
            df_pred[col] = df_pred[col].apply(lambda x: x if x in le.classes_ else 'missing')
            if 'missing' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'missing')
            df_pred[col] = le.transform(df_pred[col])
        except:
            # fallback: encode on the fly
            le = LabelEncoder()
            df_pred[col] = le.fit_transform(df_pred[col])

    # Keep only final features in order
    df_pred = df_pred[final_features]

    # ---------------------------
    # Combine Student Name
    # ---------------------------
    first_name_col = next((c for c in df_pred.columns if 'first_name' in c.lower()), None)
    last_name_col = next((c for c in df_pred.columns if 'last_name' in c.lower()), None)
    df['Student Name'] = ''
    if first_name_col or last_name_col:
        df['Student Name'] = df.get(first_name_col, '').fillna('') + ' ' + df.get(last_name_col, '').fillna('')
    else:
        df['Student Name'] = 'Unknown'

    # ---------------------------
    # Predictions
    # ---------------------------
    st.subheader("Predictions")

    model_predictions = {}
    try:
        xgb_preds = xgb_model.predict(df_pred)
        xgb_preds_labels = le_y.inverse_transform(xgb_preds)
        model_predictions['XGBoost_Predicted_Grade'] = xgb_preds_labels
    except Exception as e:
        st.error(f"XGBoost prediction error: {e}")

    try:
        cb_preds = cb_model.predict(df_pred)
        cb_preds_labels = le_y.inverse_transform(cb_preds)
        model_predictions['CatBoost_Predicted_Grade'] = cb_preds_labels
    except Exception as e:
        st.error(f"CatBoost prediction error: {e}")

    # Add predictions to df
    for col, preds in model_predictions.items():
        df[col] = preds

    # ---------------------------
    # Show Results
    # ---------------------------
    display_cols = ['Student Name'] + list(model_predictions.keys())
    st.subheader("Predictions on Uploaded Data")
    st.dataframe(df[display_cols])

    # Download results
    csv = df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )
