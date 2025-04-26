Problem Statement:

Financial institutions face significant challenges in assessing the creditworthiness of loan applicants.
Accurate credit risk prediction is crucial to minimize defaults and ensure the stability of the lending system.

This project develops a machine learning model to classify loan applicants into:

Good Credit Risk or
Bad Credit Risk
based on financial and personal data from the German Credit Dataset.

Dataset
Source: German Credit Dataset (Kaggle)

Features include:

Age, Sex, Job, Housing status

Saving accounts, Checking account

Credit amount, Duration, Purpose

Risk (Target Variable)

Objectives
Predict whether a loan applicant poses a good or bad credit risk.

Identify key features influencing credit risk.

Suggest strategies for improving the credit evaluation process.
Methods Used
1. Data Exploration & Preprocessing
Handled missing values

Encoded categorical variables

Scaled numeric features

Visualized feature correlations

2. Feature Selection
Correlation Analysis (Heatmap)

Mutual Information

Recursive Feature Elimination (RFE)

3. Model Development
Models Trained:

Decision Tree Classifier

K-Nearest Neighbors (KNN)

XGBoost Classifier

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Best model selected based on highest F1 Score.

4. Model Saving
Best Model saved as best_model.pkl

Scaler saved as scaler.pkl for consistent preprocessing during prediction.

5. Streamlit App
Simple UI where users can input features and get Credit Risk Prediction instantly.


