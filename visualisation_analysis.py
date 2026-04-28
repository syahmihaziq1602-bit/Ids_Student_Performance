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

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- VISUALIZATION 1: Correlation Heatmap ---
plt.figure(figsize=(15, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Student Performance Factors')
plt.savefig(f'{output_dir}/01_correlation_heatmap.png')
plt.close()

# --- VISUALIZATION 2: Hours Studied vs Exam Score (Regression Plot) ---
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Hours_Studied', y='Exam_Score', 
            scatter_kws={'alpha':0.3, 'color':'royalblue'}, 
            line_kws={'color':'red'})
plt.title('Impact of Study Hours on Exam Scores')
plt.savefig(f'{output_dir}/02_study_hours_vs_score.png')
plt.close()

# --- VISUALIZATION 3: Previous Scores vs Exam Score ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Previous_Scores', y='Exam_Score', hue='Motivation_Level', palette='viridis', alpha=0.6)
plt.title('Previous Scores vs Current Exam Score (Colored by Motivation)')
plt.savefig(f'{output_dir}/03_prev_score_vs_exam_score.png')
plt.close()

# --- VISUALIZATION 4: Parental Involvement (Box Plot) ---
# Assuming these are encoded, but boxplots help see the spread
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Parental_Involvement', y='Exam_Score', palette='Set2')
plt.title('Exam Score Distribution by Parental Involvement Level')
plt.savefig(f'{output_dir}/04_parental_involvement_boxplot.png')
plt.close()

# --- VISUALIZATION 5: Sleep Hours (Average Bar Chart) ---
plt.figure(figsize=(10, 6))
avg_sleep_score = df.groupby('Sleep_Hours')['Exam_Score'].mean().reset_index()
sns.barplot(data=avg_sleep_score, x='Sleep_Hours', y='Exam_Score', palette='Blues_d')
plt.title('Average Exam Score by Sleep Hours')
plt.ylabel('Mean Exam Score')
plt.savefig(f'{output_dir}/05_sleep_hours_avg_score.png')
plt.close()

print(f"All plots have been saved successfully in: {output_dir}")