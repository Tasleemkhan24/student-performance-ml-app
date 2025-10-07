import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Load models, scaler, encoder
# -----------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")
scaler = joblib.load("scaler.pkl")
le_y = joblib.load("label_encoder_y.pkl")

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.title("Student Performance Prediction")
st.write("Upload a CSV file with student data to predict grades.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of uploaded data")
    st.dataframe(df.head(10))

    # -----------------------------
    # 3. Feature Engineering
    # -----------------------------
    df_fe = df.copy()

    # Overall_Avg_Score
    cols = ['Midterm_Score_x','Final_Score_x','Assignments_Avg_x','Quizzes_Avg_x']
    df_fe['Overall_Avg_Score'] = df_fe[[c for c in cols if c in df_fe.columns]].mean(axis=1)

    # Improvement_Score
    df_fe['Improvement_Score'] = df_fe['Final_Score_x'] - df_fe['Midterm_Score_x'] if 'Final_Score_x' in df_fe.columns and 'Midterm_Score_x' in df_fe.columns else 0

    # Engagement_Score
    eng_cols = ['Participation_Score_x','Projects_Score_x','Quizzes_Avg_x']
    df_fe['Engagement_Score'] = df_fe[[c for c in eng_cols if c in df_fe.columns]].sum(axis=1)/3.0

    # Study_Efficiency
    df_fe['Study_Efficiency'] = df_fe['Total_Score_x'] / (df_fe['Study_Hours_per_Week_x'] + 1) if 'Total_Score_x' in df_fe.columns and 'Study_Hours_per_Week_x' in df_fe.columns else 0

    # Stress_Sleep_Ratio
    df_fe['Stress_Sleep_Ratio'] = df_fe['Stress_Level (1-10)_x'] / (df_fe['Sleep_Hours_per_Night_x'] + 1) if 'Stress_Level (1-10)_x' in df_fe.columns and 'Sleep_Hours_per_Night_x' in df_fe.columns else 0

    # Final_Score_Diff
    df_fe['Final_Score_Diff'] = df_fe['Final_Score_x'] - df_fe['Final_Score_y'] if 'Final_Score_y' in df_fe.columns and 'Final_Score_x' in df_fe.columns else 0

    # Attendance_Ratio
    df_fe['Attendance_Ratio'] = df_fe['Attendance (%)_x'] / (df_fe['Attendance (%)_y'] + 1) if 'Attendance (%)_x' in df_fe.columns and 'Attendance (%)_y' in df_fe.columns else 0

    # Study_Attendance
    df_fe['Study_Attendance'] = df_fe['Study_Hours_per_Week_x'] * df_fe['Attendance (%)_x'] if 'Study_Hours_per_Week_x' in df_fe.columns and 'Attendance (%)_x' in df_fe.columns else 0

    # Age_Group
    df_fe['Age_Group'] = pd.cut(df_fe['Age_x'], bins=[0,15,20,25,100], labels=['<15','15-20','20-25','25+']) if 'Age_x' in df_fe.columns else 'missing'

    # -----------------------------
    # 4. Dynamic feature list
    # -----------------------------
    # Numeric features
    numeric_feats = df_fe.select_dtypes(include=np.number).columns.tolist()

    # Categorical features (convert to string)
    cat_feats = ['First_Name_x','Last_Name_x','Email_x','Gender_x','Department_x',
                 'Grade_x','Extracurricular_Activities_x','Internet_Access_at_Home_x',
                 'Parent_Education_Level_x','Family_Income_Level_x',
                 'First_Name_y','Last_Name_y','Email_y','Gender_y','Department_y',
                 'Grade_y','Extracurricular_Activities_y','Internet_Access_at_Home_y',
                 'Parent_Education_Level_y','Family_Income_Level_y','Age_Group']
    for c in cat_feats:
        if c in df_fe.columns:
            df_fe[c] = df_fe[c].astype(str)
        else:
            df_fe[c] = 'missing'

    # Combine all features
    final_features = numeric_feats + cat_feats
    df_pred = df_fe[final_features].copy()

    # Scale numeric columns
    df_pred[numeric_feats] = scaler.transform(df_pred[numeric_feats])

    # -----------------------------
    # 5. Predictions
    # -----------------------------
    st.subheader("Predictions")
    xgb_preds = xgb_model.predict(df_pred)
    df['XGBoost_Grade'] = le_y.inverse_transform(xgb_preds)

    cb_preds = cb_model.predict(df_pred)
    df['CatBoost_Grade'] = le_y.inverse_transform(cb_preds.astype(int))

    st.dataframe(df[['Student_ID','First_Name_x','Last_Name_x','XGBoost_Grade','CatBoost_Grade']])

    # -----------------------------
    # 6. Download predictions
    # -----------------------------
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )

