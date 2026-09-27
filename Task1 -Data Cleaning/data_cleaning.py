# ======================================================
# Task 01: Data Cleaning and Visualization
# IncodeVision Internship
# ======================================================

# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PROJECT SETTINGS
# ============================================================

pd.set_option("display.max_columns", None)

os.makedirs("images", exist_ok=True)

# ------------------------------------------------------
# Load Dataset
# ------------------------------------------------------

df = pd.read_csv("dataset/train.csv")

# ------------------------------------------------------
# Display Dataset
# ------------------------------------------------------

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n")

print("=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

print("\n")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())

print("\n")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns)

print("\n")

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

# ------------------------------------------------------
# DATA CLEANING
# ------------------------------------------------------

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column because it has too many missing values
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset Shape After Cleaning:")
print(df.shape)
# ------------------------------------------------------
# SAVE CLEANED DATASET
# ------------------------------------------------------

df.to_csv("dataset/cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ------------------------------------------------------
# 1. Survival Count Plot
# ------------------------------------------------------

plt.figure(figsize=(8,5))
sns.set_style("whitegrid")

sns.countplot(data=df, x="Survived")

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 2. Age Distribution
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df["Age"], bins=30, kde=True)
sns.set_style("whitegrid")

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 3. Gender Distribution
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Sex")
sns.set_style("whitegrid")

plt.title("Gender Distribution")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 4. Passenger Class Distribution
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Pclass")
sns.set_style("whitegrid")

plt.title("Passenger Class")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 5. Survival by Gender
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Sex", hue="Survived")
sns.set_style("whitegrid")

plt.title("Survival by Gender")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 6. Survival by Passenger Class
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(data=df, x="Pclass", hue="Survived")
sns.set_style("whitegrid")

plt.title("Survival by Passenger Class")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 7. Fare Box Plot
# ------------------------------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(x=df["Fare"])
sns.set_style("whitegrid")

plt.title("Fare Box Plot")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

# ------------------------------------------------------
# 8. Correlation Heatmap
# ------------------------------------------------------

plt.figure(figsize=(8,5))
sns.set_style("whitegrid")

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "images/graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.tight_layout()
plt.show()
plt.close()

print("\n")
print("="*60)
print("TASK COMPLETED SUCCESSFULLY")
print("="*60)

print("Cleaned dataset saved inside dataset folder.")
print("Visualizations saved inside images folder.")
print("Task 01 completed successfully.")