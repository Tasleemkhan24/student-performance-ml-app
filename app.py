# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("Student Performance Prediction")
st.write("Upload a CSV file with student data to predict grades.")

# -----------------------------
# 1. Upload CSV file
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df_pred = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df_pred.head(10))

    # -----------------------------
    # 2. Load models & preprocessing
    # -----------------------------
    xgb_model = joblib.load("XGBoost_model.pkl")
    cb_model = joblib.load("CatBoost_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_y = joblib.load("label_encoder_y.pkl")  # For decoding predicted grades

    # -----------------------------
    # 3. Define numeric and categorical features
    # -----------------------------
    numeric_feats = [
        'Age_x','Attendance (%)_x','Midterm_Score_x','Assignments_Avg_x',
        'Quizzes_Avg_x','Participation_Score_x','Projects_Score_x',
        'Total_Score_x','Study_Hours_per_Week_x','Stress_Level (1-10)_x',
        'Sleep_Hours_per_Night_x','Age_y','Attendance (%)_y','Midterm_Score_y',
        'Assignments_Avg_y','Quizzes_Avg_y','Participation_Score_y',
        'Projects_Score_y','Total_Score_y','Study_Hours_per_Week_y',
        'Stress_Level (1-10)_y','Sleep_Hours_per_Night_y','Overall_Avg_Score',
        'Improvement_Score','Engagement_Score','Study_Efficiency',
        'Stress_Sleep_Ratio','Final_Score_Diff','Attendance_Ratio',
        'Study_Attendance'
    ]

    cat_feats = [c for c in df_pred.columns if c not in numeric_feats and c != 'Student_ID']

    # -----------------------------
    # 4. Fill missing values
    # -----------------------------
    for col in numeric_feats:
        if col not in df_pred.columns:
            df_pred[col] = 0

    for col in cat_feats:
        if col not in df_pred.columns:
            df_pred[col] = 'missing'
        else:
            df_pred[col] = df_pred[col].fillna('missing')

    # -----------------------------
    # 5. Scale numeric features
    # -----------------------------
    df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

    # -----------------------------
    # 6. Encode categorical features
    # -----------------------------
    for col in cat_feats:
        df_pred[col] = df_pred[col].astype(str).factorize()[0]

    # -----------------------------
    # 7. Predict using models
    # -----------------------------
    xgb_preds = xgb_model.predict(df_pred[numeric_feats + cat_feats])
    cb_preds = cb_model.predict(df_pred[numeric_feats + cat_feats])

    # Decode grades using label encoder
    xgb_preds = le_y.inverse_transform(xgb_preds)
    cb_preds = le_y.inverse_transform(cb_preds)

    # -----------------------------
    # 8. Show results
    # -----------------------------
    df_results = df_pred[['Student_ID']].copy()
    df_results['XGBoost_Grade'] = xgb_preds
    df_results['CatBoost_Grade'] = cb_preds

    st.subheader("Predicted Grades")
    st.dataframe(df_results)

    # Optional: download CSV
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predicted_grades.csv",
        mime="text/csv"
    )
