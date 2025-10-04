# student-performance-ml-app
import os
import warnings
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np_seed = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import joblib
import shap

# Aesthetic
sns.set(style="whitegrid")

# -----------------------------
# 1. Load & Merge
# -----------------------------
from google.colab import files
print("Upload: Students Performance Dataset.csv and Students_Grading_Dataset.csv")
uploaded = files.upload()  # interactively upload files

# change filenames if different
perf_fp = "Students Performance Dataset.csv"
grade_fp = "Students_Grading_Dataset_Biased.csv"

perf = pd.read_csv(perf_fp)
grade = pd.read_csv(grade_fp)

# Basic sanity: ensure Student_ID exists
assert 'Student_ID' in perf.columns and 'Student_ID' in grade.columns, "Student_ID missing"

df = perf.merge(grade, on='Student_ID', how='inner')
print(f"Merged shape: {df.shape}")

# Basic info
print("\n--- INFO ---")
display(df.info())
print("\n--- NULL counts ---")
print(df.isnull().sum().sort_values(ascending=False).head(20))
print("\n--- Describe (numeric) ---")
display(df.describe().T)

# Quick sanity checks
print("\n--- Duplicate IDs ---")
print("duplicates:", df['Student_ID'].duplicated().sum())
print("\n--- Value ranges (example) ---")
for c in ['Age_x','Study_Hours_per_Week_x','Final_Score_x']:
    if c in df.columns:
        print(c, "min/max:", df[c].min(), df[c].max())

# -----------------------------
# 2. Basic Preprocessing Helpers
# -----------------------------
def basic_clean(df):
    """Sanity cleaning: strip column names, lower where good, fix dtypes."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # convert percentage-like columns if strings with % -> numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            # remove leading/trailing spaces
            df[col] = df[col].str.strip()
    return df

df = basic_clean(df)

# -----------------------------
# 3. EDA - Visualizations
# -----------------------------
# Distribution of target (Final_Score_x)
plt.figure(figsize=(8,5))
sns.histplot(df['Final_Score_x'], bins=25, kde=True)
plt.title('Distribution: Final_Score_x')
plt.xlabel('Final Score')
plt.show()

# Boxplot by Gender (if available)
if 'Gender_x' in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Gender_x', y='Final_Score_x', data=df)
    plt.title('Final Score by Gender')
    plt.show()

# Correlation matrix for numeric features
numeric = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(12,10))
sns.heatmap(df[numeric].corr(), cmap='coolwarm', center=0)
plt.title("Numeric Feature Correlation")
plt.show()

# Pairplot for a few key features (sample for speed)
sample_cols = ['Final_Score_x','Midterm_Score_x','Total_Score_x','Study_Hours_per_Week_x']
existing = [c for c in sample_cols if c in df.columns]
if len(existing) >= 2:
    sns.pairplot(df[existing].sample(min(500, len(df))), diag_kind='kde')
    plt.show()

# -----------------------------
# 4. Outlier Detection (IQR & Z-score) + Visualization
#     (We only detect & visualize — no removal)
# -----------------------------
def detect_outliers_iqr(df, cols):
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df[cols] < (Q1 - 1.5*IQR)) | (df[cols] > (Q3 + 1.5*IQR))
    return mask

num_cols = df.select_dtypes(include=np.number).columns.tolist()
iqr_mask = detect_outliers_iqr(df, num_cols)
outlier_counts = iqr_mask.sum().sort_values(ascending=False)
print("\nOutlier counts per numeric feature (IQR method):")
display(outlier_counts.head(15))

# Visualize outliers via boxplot (subset of numeric to avoid huge plot)
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols].sample(min(1000, len(df))), orient='h')
plt.title("Boxplots (sampled rows) — visualize potential outliers")
plt.show()

# Scatter: Study Hours vs Final Score (color outliers)
if 'Study_Hours_per_Week_x' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='Study_Hours_per_Week_x', y='Final_Score_x', data=df)
    plt.title('Study Hours vs Final Score')
    plt.show()

# -----------------------------
# 5. Feature Engineering — efficient, explainable
# -----------------------------
# We create features only if source columns exist; handle missing gracefully.
df_fe = df.copy()

# Helper to safe create feature
def safe_mean(cols, out_name):
    present = [c for c in cols if c in df_fe.columns]
    if len(present) >= 1:
        df_fe[out_name] = df_fe[present].mean(axis=1)
    else:
        df_fe[out_name] = 0

# Engineered features
safe_mean(['Midterm_Score_x','Final_Score_x','Assignments_Avg_x','Quizzes_Avg_x'], 'Overall_Avg_Score')

if 'Final_Score_x' in df_fe.columns and 'Midterm_Score_x' in df_fe.columns:
    df_fe['Improvement_Score'] = df_fe['Final_Score_x'] - df_fe['Midterm_Score_x']
else:
    df_fe['Improvement_Score'] = 0

df_fe['Engagement_Score'] = df_fe[[c for c in ['Participation_Score_x','Projects_Score_x','Quizzes_Avg_x'] if c in df_fe.columns]].sum(axis=1)
# normalize engagement to mean (if non-zero)
if df_fe['Engagement_Score'].max() != 0:
    df_fe['Engagement_Score'] = df_fe['Engagement_Score'] / 3.0

# Study efficiency (avoid divide by zero)
if 'Total_Score_x' in df_fe.columns and 'Study_Hours_per_Week_x' in df_fe.columns:
    df_fe['Study_Efficiency'] = df_fe['Total_Score_x'] / (df_fe['Study_Hours_per_Week_x'] + 1)
else:
    df_fe['Study_Efficiency'] = 0

if 'Stress_Level (1-10)_x' in df_fe.columns and 'Sleep_Hours_per_Night_x' in df_fe.columns:
    df_fe['Stress_Sleep_Ratio'] = df_fe['Stress_Level (1-10)_x'] / (df_fe['Sleep_Hours_per_Night_x'] + 1)
else:
    df_fe['Stress_Sleep_Ratio'] = 0

# Comparative features if both x and y exist
if 'Final_Score_y' in df_fe.columns and 'Final_Score_x' in df_fe.columns:
    df_fe['Final_Score_Diff'] = df_fe['Final_Score_x'] - df_fe['Final_Score_y']
else:
    df_fe['Final_Score_Diff'] = 0

if 'Attendance (%)_x' in df_fe.columns and 'Attendance (%)_y' in df_fe.columns:
    df_fe['Attendance_Ratio'] = df_fe['Attendance (%)_x'] / (df_fe['Attendance (%)_y'] + 1)
else:
    df_fe['Attendance_Ratio'] = 0

# Interaction
if 'Study_Hours_per_Week_x' in df_fe.columns and 'Attendance (%)_x' in df_fe.columns:
    df_fe['Study_Attendance'] = df_fe['Study_Hours_per_Week_x'] * (df_fe['Attendance (%)_x'])
else:
    df_fe['Study_Attendance'] = 0

# Optional: Age group bucket (example)
if 'Age_x' in df_fe.columns:
    df_fe['Age_Group'] = pd.cut(df_fe['Age_x'], bins=[0,15,20,25,100], labels=['<15','15-20','20-25','25+'])
else:
    df_fe['Age_Group'] = 'Unknown'

print("\nFeature engineering complete. New columns added:")
new_cols = [c for c in df_fe.columns if c not in df.columns]
print(new_cols)

# -----------------------------
# 6. Feature selection + final feature list
# -----------------------------
# Compose final feature set: numeric features + engineered features + encoded categoricals
# Exclude identifiers & raw target
exclude = set(['Student_ID','Final_Score_x','Final_Score_y'])
all_features = [c for c in df_fe.columns if c not in exclude and c != 'Student_ID']

# For modeling we'll use numeric features and label-encoded categorical (Age_Group)
# Separate numeric and categorical
num_feats = df_fe[all_features].select_dtypes(include=np.number).columns.tolist()
cat_feats = [c for c in all_features if c not in num_feats]

print("\nNumeric features count:", len(num_feats))
print("Categorical features (to encode):", cat_feats)

# Encode categorical features with LabelEncoder (Age_Group, Gender if present)
df_model = df_fe.copy()
le_map = {}
# ✅ FIXED LOOP BELOW
for c in cat_feats:
    if pd.api.types.is_categorical_dtype(df_model[c]):
        if 'missing' not in df_model[c].cat.categories:
            df_model[c] = df_model[c].cat.add_categories(['missing'])
        df_model[c] = df_model[c].fillna('missing')
    else:
        df_model[c] = df_model[c].astype(str).fillna('missing')

    le = LabelEncoder()
    df_model[c] = le.fit_transform(df_model[c].astype(str))
    le_map[c] = le

final_features = num_feats + cat_feats
print("Final features count:", len(final_features))

df_model[final_features] = df_model[final_features].fillna(0)

# -----------------------------
# 7. Choose TASK: 'classification' or 'regression'
# -----------------------------
# Set task here:
TASK = 'classification'   # options: 'classification' or 'regression'

if TASK == 'classification':
    # create grade buckets
    def grade_category(score):
        if score >= 85: return 'A'
        elif score >= 70: return 'B'
        elif score >= 50: return 'C'
        else: return 'D'
    y = df_model['Final_Score_x'].apply(grade_category)
else:
    # regression target
    y = df_model['Final_Score_x'].astype(float)

X = df_model[final_features].copy()

# Train-test split. For classification use stratify
if TASK == 'classification':
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------------
# 8. Scaling / Pipelines
# -----------------------------
# For tree-based models scaling is not required, but we'll provide a scaler for AdaBoost and optional linear models.
scaler = StandardScaler()
# Fit scaler on training numeric portion
scaler.fit(X_train[num_feats])  # only numeric columns fitted

def scale_X(X_df):
    Xs = X_df.copy()
    Xs[num_feats] = scaler.transform(Xs[num_feats])
    return Xs

X_train_scaled = scale_X(X_train)
X_test_scaled = scale_X(X_test)

# -----------------------------
# 8.5 Encode target for classification (FIX)
# -----------------------------
if TASK == 'classification':
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    y_test = le_y.transform(y_test)
    n_classes = len(le_y.classes_)
else:
    n_classes = None

# -----------------------------
# 9. Model Training + Efficient Hyperparameter Tuning
#    Uses RandomizedSearchCV for speed. 10-30 iterations recommended.
# -----------------------------
n_iter_search = 30
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE) if TASK == 'classification' else KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

results_summary = []

# Utility functions for evaluation
def evaluate_classification(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{name} — Accuracy: {acc:.4f}, F1(weighted): {f1:.4f}")
    # use readable class names when available
    if TASK == 'classification':
        target_names = le_y.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
    else:
        print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return acc, f1

def evaluate_regression(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name} — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    # Residuals plot
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted — {name}')
    plt.show()
    return mae, rmse, r2

# ---------- Model: XGBoost ----------
if TASK == 'classification':
    # ensure multi-class objective + num_class set
    xgb = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss',
                        objective='multi:softprob', num_class=n_classes)
    param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
else:
    xgb = XGBRegressor(random_state=RANDOM_STATE)
    param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

rs_xgb = RandomizedSearchCV(
    xgb, param_distributions=param_dist, n_iter=n_iter_search,
    scoring='accuracy' if TASK=='classification' else 'neg_mean_squared_error',
    cv=cv, random_state=RANDOM_STATE, verbose=1, n_jobs=-1
)
# Fit using scaled data for XGBoost (multi-class)
if TASK == 'classification':
    rs_xgb.fit(X_train_scaled, y_train)
else:
    rs_xgb.fit(X_train, y_train)

best_xgb = rs_xgb.best_estimator_
print("\nXGBoost best params:", rs_xgb.best_params_)

# Evaluate
if TASK == 'classification':
    acc, f1 = evaluate_classification(best_xgb, X_test_scaled, y_test, name="XGBoost")
    results_summary.append(('XGBoost', acc, f1, rs_xgb.best_params_))
else:
    mae, rmse, r2 = evaluate_regression(best_xgb, X_test_scaled, y_test, name="XGBoost")
    results_summary.append(('XGBoost', mae, rmse, r2, rs_xgb.best_params_))

# ---------- Model: CatBoost ----------
if TASK == 'classification':
    cb = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    param_dist = {
        'iterations': [200, 400, 800],
        'depth': [4,6,8],
        'learning_rate': [0.01, 0.03, 0.1],
        'l2_leaf_reg': [1,3,5,7]
    }
else:
    cb = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
    param_dist = {
        'iterations': [200, 400, 800],
        'depth': [4,6,8],
        'learning_rate': [0.01, 0.03, 0.1],
        'l2_leaf_reg': [1,3,5,7]
    }

rs_cb = RandomizedSearchCV(
    cb, param_distributions=param_dist, n_iter=12,
    scoring='accuracy' if TASK=='classification' else 'neg_mean_squared_error',
    cv=cv, random_state=RANDOM_STATE, verbose=1, n_jobs=-1
)
# CatBoost can accept pandas DataFrames directly. Use unscaled or scaled as you prefer.
rs_cb.fit(X_train, y_train)
best_cb = rs_cb.best_estimator_
print("\nCatBoost best params:", rs_cb.best_params_)

# Evaluate
if TASK == 'classification':
    acc, f1 = evaluate_classification(best_cb, X_test, y_test, name="CatBoost")
    results_summary.append(('CatBoost', acc, f1, rs_cb.best_params_))
else:
    mae, rmse, r2 = evaluate_regression(best_cb, X_test, y_test, name="CatBoost")
    results_summary.append(('CatBoost', mae, rmse, r2, rs_cb.best_params_))

# ---------- Model: AdaBoost ----------
if TASK == 'classification':
    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    param_dist = {
        'n_estimators': [50,100,200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
else:
    ada = AdaBoostRegressor(random_state=RANDOM_STATE)
    param_dist = {
        'n_estimators': [50,100,200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }

rs_ada = RandomizedSearchCV(
    ada, param_distributions=param_dist, n_iter=10,
    scoring='accuracy' if TASK=='classification' else 'neg_mean_squared_error',
    cv=cv, random_state=RANDOM_STATE, verbose=1, n_jobs=-1
)
# AdaBoost prefers scaled features for some base estimators
rs_ada.fit(X_train_scaled, y_train)
best_ada = rs_ada.best_estimator_
print("\nAdaBoost best params:", rs_ada.best_params_)

# Evaluate
if TASK == 'classification':
    acc, f1 = evaluate_classification(best_ada, X_test_scaled, y_test, name="AdaBoost")
    results_summary.append(('AdaBoost', acc, f1, rs_ada.best_params_))
else:
    mae, rmse, r2 = evaluate_regression(best_ada, X_test_scaled, y_test, name="AdaBoost")
    results_summary.append(('AdaBoost', mae, rmse, r2, rs_ada.best_params_))

# -----------------------------
# 10. Feature importance & SHAP interpretability (for tree models)
# -----------------------------
print("\n=== Feature importance (XGBoost) ===")
if TASK == 'classification' or TASK == 'regression':
    try:
        fi = best_xgb.get_booster().get_score(importance_type='weight')
        fi_df = pd.DataFrame.from_dict(fi, orient='index', columns=['importance']).sort_values('importance', ascending=False)
        display(fi_df.head(20))
    except Exception as e:
        print("Could not extract XGBoost feature importance:", e)

# SHAP (explainability) — might be slow on large datasets
try:
    explainer = shap.TreeExplainer(best_xgb)
    X_shap = X_train_scaled if TASK=='classification' else X_train_scaled
    # SHAP can expect numpy array or DataFrame; pass sample for speed
    shap_values = explainer.shap_values(X_shap[:500])
    shap.summary_plot(shap_values, X_shap[:500], feature_names=final_features, plot_type="bar", show=True)
except Exception as e:
    print("SHAP plotting skipped (reason):", e)

# -----------------------------
# 11. Save models & scaler
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_xgb, "models/best_xgb.pkl")
joblib.dump(best_cb, "models/best_catboost.pkl")
joblib.dump(best_ada, "models/best_adaboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")
# save target encoder to map predictions back to original labels
if TASK == 'classification':
    joblib.dump(le_y, "models/label_encoder_y.pkl")
print("Saved models to /models")

# -----------------------------
# 12. Final results table (compact)
# -----------------------------
if TASK == 'classification':
    summary_df = pd.DataFrame([
        {'model': r[0], 'accuracy': r[1], 'f1_weighted': r[2], 'best_params': r[3]} for r in results_summary
    ])
else:
    summary_df = pd.DataFrame([
        {'model': r[0], 'mae': r[1], 'rmse': r[2], 'r2': r[3], 'best_params': r[4]} for r in results_summary
    ])

display(summary_df)
print("Pipeline complete ✅")
