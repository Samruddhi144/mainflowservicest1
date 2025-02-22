import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('student-mat.csv')

data.head()
missing_values = data.isnull().sum()

data_types = data.dtypes

data_shape = data.shape

data.fillna(data.median(numeric_only=True), inplace=True)

data.drop_duplicates(inplace=True)

# 1. Average score in math (G3)
avg_score = data['G3'].mean()

# 2. Count of students scoring above 15 in final grade (G3)
high_scorers = data[data['G3'] > 15].shape[0]

# 3. Correlation between study time and final grade
correlation = data['studytime'].corr(data['G3'])

# 4. Average final grade by gender
avg_grade_gender = data.groupby('sex')['G3'].mean()

# Step 5: Data Visualization
# Histogram of final grades
plt.figure(figsize=(8, 5))
plt.hist(data['G3'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot between study time and final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x='studytime', y='G3', data=data, color='green')
plt.title('Study Time vs Final Grade')
plt.xlabel('Study Time (hours per week)')
plt.ylabel('Final Grade (G3)')
plt.show()

# Bar chart comparing average scores by gender
plt.figure(figsize=(6, 4))
avg_grade_gender.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Average Final Grade by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Final Grade (G3)')
plt.xticks(rotation=0)
plt.show()

# Summary of findings
print("Missing Values:\n", missing_values)
print("\nData Types:\n", data_types)
print("\nDataset Size:", data_shape)
print("\nAverage Final Grade (G3):", avg_score)
print("\nNumber of Students Scoring Above 15 in G3:", high_scorers)
print("\nCorrelation between Study Time and Final Grade:", correlation)
print("\nAverage Final Grade by Gender:\n", avg_grade_gender)
