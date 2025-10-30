
"""
Task 2 - Exploratory Data Analysis (EDA) | Titanic Dataset
Performs cleaning, feature engineering, and statistical exploration.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    return sns.load_dataset("titanic")

def clean_data(df):
    df = df.dropna(subset=["age", "embarked", "fare"])
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["alone"] = df["alone"].astype(int)
    return df

def analyze(df):
    print("📊 Dataset Overview:")
    print(df.describe())
    print("\n🎯 Missing values:")
    print(df.isnull().sum())
    print("\n🔗 Correlation Matrix:")
    print(df.corr(numeric_only=True))

def visualize(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap - Titanic Data")
    plt.savefig("task2_correlation_heatmap.png")

    plt.figure(figsize=(8, 6))
    sns.histplot(df["age"], bins=30, kde=True, color="blue")
    plt.title("Age Distribution")
    plt.savefig("task2_age_distribution.png")

if __name__ == "__main__":
    data = load_data()
    clean = clean_data(data)
    analyze(clean)
    visualize(clean)
    clean.to_csv("task2_cleaned_titanic.csv", index=False)
    print("✅ Cleaned Titanic dataset and plots saved.")
