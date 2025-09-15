# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Covid-Data.csv')

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

df.loc[df['SEX'] == 2, 'PREGNANT'] = 2

updated_rows = df[df['SEX'] == 2]
print("Updated rows where SEX is 2:\n", updated_rows)

unique_sex_values = df['PREGNANT'].unique()
print("Unique values in the SEX column:", unique_sex_values)

df = df[~df['INTUBED'].isin([97, 98, 99]) & ~df['PREGNANT'].isin([97, 98, 99])]

print("Number of rows after removal:", len(df))

df_cleaned_copy = df.copy()

df_cleaned_copy.loc[:, 'DATE_DIED'] = df_cleaned_copy['DATE_DIED'].apply(lambda x: 2 if x == '9999-99-99' else 1)

print("Number of rows in df_cleaned_copy:", len(df_cleaned_copy))

columns_to_clean = df_cleaned_copy.columns[df_cleaned_copy.columns != 'AGE']

# Replace 97, 98, 99 with NaN in specified columns
df_cleaned_copy[columns_to_clean] = df_cleaned_copy[columns_to_clean].replace([97, 98, 99], np.nan)

# Optionally, drop rows with any NaN values (if you want to remove those rows)
df_cleaned_copy = df_cleaned_copy.dropna()

# Display the modified DataFrame
print("DataFrame after removing values 97, 98, 99 from all columns except AGE:\n", df_cleaned_copy.head())

# Print the number of rows after cleaning
print("Number of rows after cleaning:", df_cleaned_copy.shape[0])

df_cleaned_copy = df_cleaned_copy.drop(columns=['PATIENT_TYPE'])

# Optionally, print the updated DataFrame to confirm the column is removed
print("Updated DataFrame after dropping 'PATIENT_TYPE':\n", df_cleaned_copy.head())

numerical_columns = df_cleaned_copy.select_dtypes(include=['float64', 'int64']).columns
print(numerical_columns)

for column in numerical_columns:
    print(f"Column: {column}")
    print(f"  Unique values: {df_cleaned_copy[column].unique()}")
    print(f"  Min: {df_cleaned_copy[column].min()}, Max: {df_cleaned_copy[column].max()}")
    print(f"  Mean: {df_cleaned_copy[column].mean()}, Std: {df_cleaned_copy[column].std()}")
    print(f"  Skewness: {df_cleaned_copy[column].skew()}")
    print("-" * 40)

count_class_1 = df_cleaned_copy[df_cleaned_copy['DATE_DIED'] == 1].shape[0]
count_class_2 = df_cleaned_copy[df_cleaned_copy['DATE_DIED'] == 2].shape[0]

print(f"Number of deceased patients (1): {count_class_1}")
print(f"Number of alive patients (2): {count_class_2}")

df_alive = df_cleaned_copy[df_cleaned_copy['DATE_DIED'] == 2]
df_deceased = df_cleaned_copy[df_cleaned_copy['DATE_DIED'] == 1]

df_alive_downsampled = df_alive.sample(count_class_1, random_state=42)

df_balanced = pd.concat([df_alive_downsampled, df_deceased])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced dataset class counts:")
print(df_balanced['DATE_DIED'].value_counts())

X = df_cleaned_copy.drop(columns=['DATE_DIED'])
y = df_cleaned_copy['DATE_DIED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate additional regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

from sklearn.model_selection import learning_curve

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# Calculate mean and standard deviation for training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot the learning curve
plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()

# Calculate the correlation matrix
correlation_matrix = df_cleaned_copy.corr()

# Plot the heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

y_pred_rounded = y_pred.round().astype(int)

accuracy = accuracy_score(y_test, y_pred_rounded)
conf_matrix = confusion_matrix(y_test, y_pred_rounded)
class_report = classification_report(y_test, y_pred_rounded)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)