# app.py
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load models
# -----------------------------
xgb_model = joblib.load("XGBoost_model.pkl")
cb_model = joblib.load("CatBoost_model.pkl")

# -----------------------------
# 2. Final features used in training
# -----------------------------
final_features = [
    'Age_x','Attendance (%)_x','Midterm_Score_x','Assignments_Avg_x',
    'Quizzes_Avg_x','Participation_Score_x','Projects_Score_x',
    'Total_Score_x','Study_Hours_per_Week_x','Stress_Level (1-10)_x',
    'Sleep_Hours_per_Night_x','Age_y','Attendance (%)_y','Midterm_Score_y',
    'Assignments_Avg_y','Quizzes_Avg_y','Participation_Score_y',
    'Projects_Score_y','Total_Score_y','Study_Hours_per_Week_y',
    'Stress_Level (1-10)_y','Sleep_Hours_per_Night_y','Overall_Avg_Score',
    'Improvement_Score','Engagement_Score','Study_Efficiency',
    'Stress_Sleep_Ratio','Final_Score_Diff','Attendance_Ratio',
    'Study_Attendance','First_Name_x','Last_Name_x','Email_x','Gender_x',
    'Department_x','Grade_x','Extracurricular_Activities_x',
    'Internet_Access_at_Home_x','Parent_Education_Level_x',
    'Family_Income_Level_x','First_Name_y','Last_Name_y','Email_y','Gender_y',
    'Department_y','Grade_y','Extracurricular_Activities_y',
    'Internet_Access_at_Home_y','Parent_Education_Level_y',
    'Family_Income_Level_y','Age_Group'
]

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.write("Upload a CSV of students to predict grades using XGBoost or CatBoost.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # -----------------------------
    # 4. Preprocessing
    # -----------------------------
    df_pred = df.copy()

    # Ensure all final_features exist
    for col in final_features:
        if col not in df_pred.columns:
            df_pred[col] = 0  # fill missing columns

    df_pred = df_pred[final_features].copy()

    # Convert categorical columns to string for tree models
    cat_cols = ['First_Name_x','Last_Name_x','Email_x','Gender_x','Department_x','Grade_x',
                'Extracurricular_Activities_x','Internet_Access_at_Home_x','Parent_Education_Level_x',
                'Family_Income_Level_x','First_Name_y','Last_Name_y','Email_y','Gender_y',
                'Department_y','Grade_y','Extracurricular_Activities_y',
                'Internet_Access_at_Home_y','Parent_Education_Level_y','Family_Income_Level_y','Age_Group']
    for c in cat_cols:
        if c in df_pred.columns:
            df_pred[c] = df_pred[c].astype(str).fillna('missing')

    # -----------------------------
    # 5. Model selection
    # -----------------------------
    model_choice = st.radio("Choose model:", ("XGBoost", "CatBoost"))

    if model_choice == "XGBoost":
        preds = xgb_model.predict(df_pred)
    else:
        preds = cb_model.predict(df_pred)

    # -----------------------------
    # 6. Display results
    # -----------------------------
    results = pd.DataFrame({
        "First_Name": df_pred['First_Name_x'],
        "Last_Name": df_pred['Last_Name_x'],
        "Predicted_Grade": preds
    })

    st.write("✅ Predictions:")
    st.dataframe(results)

    st.download_button(
        "📥 Download Results as CSV",
        results.to_csv(index=False),
        "predicted_grades.csv",
        "text/csv"
    )
