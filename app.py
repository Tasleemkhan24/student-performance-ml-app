import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load trained models
# -----------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")

# List of features used during training (replace with your exact features)
final_features = [
    'Age_x','Attendance (%)_x','Midterm_Score_x','Assignments_Avg_x','Quizzes_Avg_x',
    'Participation_Score_x','Projects_Score_x','Total_Score_x','Study_Hours_per_Week_x',
    'Stress_Level (1-10)_x','Sleep_Hours_per_Night_x','Age_y','Attendance (%)_y','Midterm_Score_y',
    'Assignments_Avg_y','Quizzes_Avg_y','Participation_Score_y','Projects_Score_y','Total_Score_y',
    'Study_Hours_per_Week_y','Stress_Level (1-10)_y','Sleep_Hours_per_Night_y','Overall_Avg_Score',
    'Improvement_Score','Engagement_Score','Study_Efficiency','Stress_Sleep_Ratio','Final_Score_Diff',
    'Attendance_Ratio','Study_Attendance','First_Name_x','Last_Name_x','Email_x','Gender_x','Department_x',
    'Grade_x','Extracurricular_Activities_x','Internet_Access_at_Home_x','Parent_Education_Level_x',
    'Family_Income_Level_x','First_Name_y','Last_Name_y','Email_y','Gender_y','Department_y','Grade_y',
    'Extracurricular_Activities_y','Internet_Access_at_Home_y','Parent_Education_Level_y','Family_Income_Level_y',
    'Age_Group'
]

# Categorical features (must match training)
cat_feats = [
    'First_Name_x','Last_Name_x','Email_x','Gender_x','Department_x','Grade_x',
    'Extracurricular_Activities_x','Internet_Access_at_Home_x','Parent_Education_Level_x',
    'Family_Income_Level_x','First_Name_y','Last_Name_y','Email_y','Gender_y','Department_y','Grade_y',
    'Extracurricular_Activities_y','Internet_Access_at_Home_y','Parent_Education_Level_y',
    'Family_Income_Level_y','Age_Group'
]

# -----------------------------
# 2. Upload dataset
# -----------------------------
st.title("Student Performance Prediction (XGBoost + CatBoost)")
uploaded_file = st.file_uploader("Upload CSV file for prediction", type=["csv"])

if uploaded_file:
    df_pred = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df_pred.head())

    # -----------------------------
    # 3. Keep only trained features
    # -----------------------------
    df_pred_model = df_pred[final_features].copy()

    # -----------------------------
    # 4. Handle categorical features
    # -----------------------------
    for c in cat_feats:
        if c in df_pred_model.columns:
            df_pred_model[c] = df_pred_model[c].astype(str).fillna('missing')

    # -----------------------------
    # 5. Predict with XGBoost
    # -----------------------------
    st.subheader("XGBoost Predictions")
    try:
        preds_xgb = xgb_model.predict(df_pred_model)
        st.write(preds_xgb)
    except Exception as e:
        st.error(f"XGBoost prediction failed: {e}")

    # -----------------------------
    # 6. Predict with CatBoost
    # -----------------------------
    st.subheader("CatBoost Predictions")
    try:
        preds_cb = cb_model.predict(df_pred_model)
        st.write(preds_cb)
    except Exception as e:
        st.error(f"CatBoost prediction failed: {e}")
