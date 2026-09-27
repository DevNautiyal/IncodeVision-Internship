# 📊 Customer Churn Prediction

## Overview

This project predicts whether a telecom customer is likely to leave (churn) or stay using Machine Learning. It includes data cleaning, exploratory data analysis, model training, evaluation, and a Streamlit web application for real-time prediction.

---

## Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Train-Test Split
- Logistic Regression
- Decision Tree
- Random Forest
- Model Evaluation
- Streamlit Web Application
- Saved Machine Learning Model

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## Dataset

Telco Customer Churn Dataset

---

## Project Structure

```
Task2 -Customer Churn/
│
├── app.py
├── churn_prediction.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── cleaned_customer_churn.csv
│
├── model/
│   ├── churn_model.joblib
│   └── encoders.joblib
│
└── images/
```

---

## How to Run

1. Create a virtual environment.
2. Install the required libraries.
3. Run:

```
streamlit run app.py
```

---

## Model Accuracy

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 79.57% |
| Decision Tree | 73.59% |
| Random Forest | 78.22% |

---

## Author

Dev