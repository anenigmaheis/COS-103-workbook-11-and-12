"""Eleven: 
Load the Iris dataset and perform exploratory data analysis using Python. Visualize the data using various plots and calculate summary statistics."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")

print(df.head())
print("Missing values:", df.isna().sum().sum())

# Summary statistics
print(df.describe())
print(df.groupby("species").mean())

# Visualizations
df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x="species", y="petal length (cm)")
plt.show()

sns.pairplot(df, hue="species")
plt.show()

sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm")
plt.show()
