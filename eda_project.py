import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("EXPLORATORY DATA ANALYSIS")
print("=========================")

print("First 5 Rows:")
print(df.head())

print("Dataset Shape:")
print(df.shape)

print("Statistical Summary:")
print(df.describe())

print("Missing Values:")
print(df.isnull().sum())

correlation = df.corr()

print("Correlation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    df["sepal length (cm)"],
    df["petal length (cm)"]
)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df["sepal length (cm)"], bins=10)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Sepal Length")
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(df.values)
plt.xticks(
    [1, 2, 3, 4],
    df.columns,
    rotation=20
)
plt.title("Box Plot of Iris Features")
plt.show()

print("EDA PROJECT COMPLETED SUCCESSFULLY")
