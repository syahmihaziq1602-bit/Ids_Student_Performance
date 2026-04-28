import pandas as pd
import numpy as np

# 1. LOAD DATASET
file_path = 'StudentPerformanceFactors (2).csv'
df = pd.read_csv(file_path)

# 2. DROP UNNECESSARY COLUMNS
df = df.drop(columns=['Distance_from_Home', 'Parental_Education_Level'])

# 3. DATA CLEANING
df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df.loc[df['Exam_Score'] > 100, 'Exam_Score'] = 100
df = df.drop_duplicates()

# 4. FEATURE ENGINEERING
df['Total_Learning_Effort'] = df['Hours_Studied'] + df['Tutoring_Sessions']

# 5. OUTLIER REMOVAL
mean_hours = df['Hours_Studied'].mean()
std_hours = df['Hours_Studied'].std()
df = df[(df['Hours_Studied'] <= mean_hours + 3*std_hours) & (df['Hours_Studied'] >= mean_hours - 3*std_hours)]

# 6. DATA ENCODING
binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Public': 1, 'Private': 0}
for col in ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'Gender', 'School_Type']:
    df[col] = df[col].map(binary_map)

scale_map = {'Low': 1, 'Medium': 2, 'High': 3}
for col in ['Motivation_Level', 'Parental_Involvement', 'Access_to_Resources', 'Family_Income', 'Teacher_Quality']:
    df[col] = df[col].map(scale_map)

df['Peer_Influence'] = df['Peer_Influence'].map({'Negative': 1, 'Neutral': 2, 'Positive': 3})

# 7. DATA NORMALIZATION (Min-Max Scaling)
# This scales all numerical values to a range between 0 and 1
cols_to_scale = df.select_dtypes(include=['int64', 'float64']).columns
for col in cols_to_scale:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    # Scale and then round to 3 decimal places
    df[col] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())).round(3)

# 8. SAVE THE ML-READY DATASET
df.to_csv('Student_Performance_fin.csv', index=False)

print("\n--- ML-READY CLEANING COMPLETE ---")
print(f"Total Columns: {len(df.columns)}")
print("All numeric features have been normalized (0 to 1).")
print("File Saved as: Student_Performance_Fin.csv")