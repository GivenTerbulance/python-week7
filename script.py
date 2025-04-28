# Task 1: Load and Explore the Dataset
# Step 1: Import pandas library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Step 2 : loading data from iris 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset using .head() to inspect the data.
print(df.head())

# Exploring the structure of the dataset by checking the data types and any missing values.
print(df.info())


print(df.isnull().sum())  

# Cleaning the dataset by either filling or dropping any missing values.
df_cleaned = df.dropna() 


print(df_cleaned.head())

#Task 2: Basic Data Analysis

# Step 1: Compute basic statistics for numerical columns
# The .describe() function provides statistics like mean, std, min, 25%, 50%, 75%, and max
stats = df_cleaned.describe()
print(stats)

# Step 2: Group by the 'species' column and compute the mean of numerical columns for each group
grouped_by_species = df_cleaned.groupby('species').mean()
print(grouped_by_species)

# Step 3: Identify patterns or interesting findings
# We can analyze the results of the descriptive statistics and groupings to find insights.
# For example, looking at the mean of numerical columns for each species could show us differences.
# Importing the necessary libraries


# Step 1: Line chart (Simulated example with an index as time)
# We can simulate a "time" column using the index for this example
df_cleaned['time'] = range(len(df_cleaned))  # Simulate a time series

plt.figure(figsize=(10, 6))
plt.plot(df_cleaned['time'], df_cleaned['sepal_length'], label='Sepal Length', color='b')
plt.title('Sepal Length Trend Over Time')
plt.xlabel('Time')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# Step 2: Bar chart (Average Petal Length per Species)
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal_length', data=df_cleaned, palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Step 3: Histogram (Distribution of Sepal Length)
plt.figure(figsize=(8, 6))
sns.histplot(df_cleaned['sepal_length'], kde=True, color='green', bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Step 4: Scatter plot (Sepal Length vs. Petal Length)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df_cleaned, palette='Set1')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
