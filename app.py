import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# 1. Load models & utilities
# -----------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Student Performance Prediction")

uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file:
    df_pred = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_pred.head())

    # -----------------------------
    # 3. Prepare features
    # -----------------------------
    # Infer features used by XGBoost from its booster
    xgb_features = xgb_model.get_booster().feature_names
    df_model = df_pred.copy()
    
    # Keep only columns that XGBoost was trained on
    df_model = df_model[xgb_features]

    # Separate numeric and categorical
    numeric_feats = df_model.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in df_model.columns if c not in numeric_feats]

    # Handle categorical columns
    for c in cat_feats:
        df_model[c] = df_model[c].astype(str).fillna('missing')

    # Scale numeric columns
    for c in numeric_feats:
        df_model[c] = scaler.transform(df_model[[c]])

    # -----------------------------
    # 4. Predictions
    # -----------------------------
    st.subheader("Predictions")

    try:
        preds_xgb = xgb_model.predict(df_model)
        preds_cb = cb_model.predict(df_model)

        # Map numeric predictions back to original labels
        preds_xgb_label = le_y.inverse_transform(preds_xgb)
        preds_cb_label = le_y.inverse_transform(preds_cb)

        # Show only student name + prediction
        if 'First_Name_x' in df_pred.columns and 'Last_Name_x' in df_pred.columns:
            result_df = pd.DataFrame({
                'First_Name': df_pred['First_Name_x'],
                'Last_Name': df_pred['Last_Name_x'],
                'XGBoost_Prediction': preds_xgb_label,
                'CatBoost_Prediction': preds_cb_label
            })
        else:
            result_df = pd.DataFrame({
                'XGBoost_Prediction': preds_xgb_label,
                'CatBoost_Prediction': preds_cb_label
            })

        st.dataframe(result_df)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
