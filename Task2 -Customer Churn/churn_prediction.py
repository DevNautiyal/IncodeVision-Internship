# ============================================================
# Task 04: Customer Churn Prediction
# Machine Learning Internship - IncodeVision
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ------------------------------------------------------------
# Display First 5 Rows
# ------------------------------------------------------------

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

# ------------------------------------------------------------
# Display Last 5 Rows
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

# ------------------------------------------------------------
# Dataset Information
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())

# ------------------------------------------------------------
# Dataset Shape
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

# ------------------------------------------------------------
# Column Names
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns)

# ------------------------------------------------------------
# Statistical Summary
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# ------------------------------------------------------------
# Duplicate Rows
# ------------------------------------------------------------

print("\n")
print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

# ============================================================
# DATA CLEANING
# ============================================================

print("\n")
print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove customerID column (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values again
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Check duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

print("Shape after cleaning:", df.shape)

# Save cleaned dataset
df.to_csv("dataset/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\n")
print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Create images folder if it doesn't exist
import os
os.makedirs("images", exist_ok=True)

# ------------------------------------------------------------
# 1. Churn Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.savefig("images/churn_distribution.png")
plt.show()

# ------------------------------------------------------------
# 2. Gender Distribution
# ------------------------------------------------------------

plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.savefig("images/gender_vs_churn.png")
plt.show()

# ------------------------------------------------------------
# 3. Contract Type
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.savefig("images/contract_vs_churn.png")
plt.show()

# ------------------------------------------------------------
# 4. Internet Service
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Internet Service vs Churn")
plt.savefig("images/internet_service.png")
plt.show()

# ------------------------------------------------------------
# 5. Tenure Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Tenure Distribution")
plt.savefig("images/tenure_distribution.png")
plt.show()

# ------------------------------------------------------------
# 6. Monthly Charges
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.savefig("images/monthly_charges.png")
plt.show()

# ------------------------------------------------------------
# 7. Total Charges
# ------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["TotalCharges"], bins=30, kde=True)
plt.title("Total Charges Distribution")
plt.savefig("images/total_charges.png")
plt.show()

print("\nEDA completed successfully!")

# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\n")
print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder

import joblib

encoders = {}

for col in df.columns:
    if df[col].dtype != "int64" and df[col].dtype != "float64":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

joblib.dump(encoders, "model/encoders.joblib")
print(df.head())
print(df.dtypes)

print("\nUnique values in gender:")
print(df["gender"].unique())

print("\nColumns still having object datatype:")
print(df.select_dtypes(include=["object"]).columns)
# ============================================================
# TRAIN TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nColumns with object/string datatype:")
print(X.select_dtypes(include=["object", "string"]).columns)

print("\nData types:")
print(X.dtypes)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print("\nModels trained successfully!")

# ============================================================
# MODEL EVALUATION
# ============================================================

from sklearn.metrics import accuracy_score

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nLogistic Regression Accuracy :", round(lr_acc*100,2), "%")
print("Decision Tree Accuracy       :", round(dt_acc*100,2), "%")
print("Random Forest Accuracy       :", round(rf_acc*100,2), "%")

# ============================================================
# SAVE BEST MODEL
# ============================================================

import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(rf, "model/churn_model.joblib")

print("\nRandom Forest model saved successfully!")