import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Setup Directory
output_dir = 'visualizations_analysis'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Load Dataset
df = pd.read_csv('Student_Performance_fin.csv')

# --- DATA MAPPING: Reverting to Original Ranges ---
# Hours Studied: Min 1, Max 44 (Range 43)
df['Hours_Studied_Actual'] = (df['Hours_Studied'] * 43 + 1).round()

# Sleep Hours: Min 4, Max 10 (Range 6)
df['Sleep_Hours_Actual'] = (df['Sleep_Hours'] * 6 + 4).round().astype(int)

# Previous Scores: Min 50, Max 100 (Range 50)
df['Previous_Scores_Actual'] = (df['Previous_Scores'] * 50 + 50).round()

# Exam Score: Min 55, Max 100 (Range 45)
df['Exam_Score_Actual'] = (df['Exam_Score'] * 45 + 55).round()

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- VISUALIZATION 1: Correlation Heatmap (Tanpa column 'Actual') ---
plt.figure(figsize=(15, 10))
original_cols = [col for col in df.columns if 'Actual' not in col]
corr_matrix = df[original_cols].select_dtypes(include=['number']).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Student Performance Factors')
plt.savefig(f'{output_dir}/01_correlation_heatmap.png')
plt.close()

# --- VISUALIZATION 2: Hours Studied vs Exam Score (Original Range) ---
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Hours_Studied_Actual', y='Exam_Score_Actual', 
            scatter_kws={'alpha':0.3, 'color':'royalblue'}, 
            line_kws={'color':'red'})
plt.title('Impact of Study Hours on Exam Scores')
plt.xlabel('Hours Studied (1 - 44)')
plt.ylabel('Exam Score (55 - 100)')
plt.savefig(f'{output_dir}/02_study_hours_vs_score.png')
plt.close()

# --- VISUALIZATION 3: Previous Scores vs Exam Score (Original Range) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Previous_Scores_Actual', y='Exam_Score_Actual', 
                hue='Motivation_Level', palette='viridis', alpha=0.6)
plt.title('Previous Scores vs Current Exam Score')
plt.xlabel('Previous Scores (50 - 100)')
plt.ylabel('Exam Score (55 - 100)')
plt.savefig(f'{output_dir}/03_prev_score_vs_exam_score.png')
plt.close()

# --- VISUALIZATION 4: Parental Involvement Box Plot (Original Y Range) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Parental_Involvement', y='Exam_Score_Actual', palette='Set2')
plt.title('Exam Score Distribution by Parental Involvement Level')
plt.ylabel('Exam Score (55 - 100)')
plt.savefig(f'{output_dir}/04_parental_involvement_boxplot.png')
plt.close()

# --- VISUALIZATION 5: Sleep Hours Average Bar Chart (Original X & Y Ranges) ---
plt.figure(figsize=(10, 6))
# Calculate mean using original exam scores
avg_sleep_score = df.groupby('Sleep_Hours_Actual')['Exam_Score_Actual'].mean().reset_index()
sns.barplot(data=avg_sleep_score, x='Sleep_Hours_Actual', y='Exam_Score_Actual', palette='Blues_d')

plt.title('Average Exam Score by Sleep Hours')
plt.xlabel('Sleep Hours (4 - 10)')
plt.ylabel('Mean Exam Score (55 - 100)')
plt.savefig(f'{output_dir}/05_sleep_hours_avg_score.png')
plt.close()

print(f"All plots have been saved successfully in: {output_dir}")