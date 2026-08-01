import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def preprocess_dataset_1():
    print("Preprocessing Dataset 1...")
    df = pd.read_csv(os.path.join(DATA_DIR, "Student_Performance.csv"))
    
    # 1. Encode Extracurricular Activities
    le = LabelEncoder()
    df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])
    
    # Features and target
    X = df.drop(columns=['Performance Index'])
    y = df['Performance Index']
    
    # Save raw columns for feature selection
    feature_cols = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Scale features
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Scale target (StandardScaler for Target is required to reproduce paper's RMSE/MAE)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Prepare feature selection subset (Previous Scores and Hours Studied are index 1 and 0 in X)
    idx_prev_scores = feature_cols.index('Previous Scores')
    idx_hours_studied = feature_cols.index('Hours Studied')
    
    X_train_fs = X_train_scaled[:, [idx_hours_studied, idx_prev_scores]]
    X_test_fs = X_test_scaled[:, [idx_hours_studied, idx_prev_scores]]
    
    # Save preprocessing objects
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    with open(os.path.join(MODELS_DIR, "dataset_1_preprocessors.pkl"), "wb") as f:
        pickle.dump({
            'label_encoder': le,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols,
            'fs_cols': ['Hours Studied', 'Previous Scores'],
            'fs_indices': [idx_hours_studied, idx_prev_scores]
        }, f)
        
    print("Dataset 1 preprocessed successfully.")
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_train_fs, X_test_fs, y_train, y_test

def preprocess_dataset_2():
    print("Preprocessing Dataset 2...")
    df = pd.read_csv(os.path.join(DATA_DIR, "StudentPerformanceFactors.csv"))
    
    # 1. Outliers: replace 101 with 100 in Exam_Score
    df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100
    
    # 2. Impute missing values
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Exam_Score' in numerical_cols:
        numerical_cols.remove('Exam_Score')
        
    modes = {}
    means = {}
    
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        modes[col] = mode_val
        df[col] = df[col].fillna(mode_val)
        
    for col in numerical_cols:
        mean_val = df[col].mean()
        means[col] = mean_val
        df[col] = df[col].fillna(mean_val)
        
    # 3. Custom Ordinal mappings to replicate scientific report results
    ordinal_mappings = {
        'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
        'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
        'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
        'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
        'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
        'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
        'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
        'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1},
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Learning_Disabilities': {'No': 0, 'Yes': 1},
        'School_Type': {'Public': 0, 'Private': 1},
        'Gender': {'Female': 0, 'Male': 1}
    }
    
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping)
        
    # Features and target
    X = df.drop(columns=['Exam_Score'])
    y = df['Exam_Score']
    
    feature_cols = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Scale features
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Prepare feature selection subset (Attendance, Hours_Studied, Access_to_Resources, Previous_Scores)
    idx_attendance = feature_cols.index('Attendance')
    idx_hours_studied = feature_cols.index('Hours_Studied')
    idx_access = feature_cols.index('Access_to_Resources')
    idx_prev_scores = feature_cols.index('Previous_Scores')
    
    X_train_fs = X_train_scaled[:, [idx_attendance, idx_hours_studied, idx_access, idx_prev_scores]]
    X_test_fs = X_test_scaled[:, [idx_attendance, idx_hours_studied, idx_access, idx_prev_scores]]
    
    # Save preprocessing objects
    with open(os.path.join(MODELS_DIR, "dataset_2_preprocessors.pkl"), "wb") as f:
        pickle.dump({
            'ordinal_mappings': ordinal_mappings,
            'scaler_x': scaler_x,
            'feature_cols': feature_cols,
            'fs_cols': ['Attendance', 'Hours_Studied', 'Access_to_Resources', 'Previous_Scores'],
            'fs_indices': [idx_attendance, idx_hours_studied, idx_access, idx_prev_scores],
            'modes': modes,
            'means': means
        }, f)
        
    print("Dataset 2 preprocessed successfully.")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X_train_fs, X_test_fs

if __name__ == "__main__":
    preprocess_dataset_1()
    preprocess_dataset_2()
