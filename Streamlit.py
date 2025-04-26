import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and encoder if available
@st.cache_data
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("german_credit_data.csv")

# UI: App title and description
st.title("💳 Credit Risk Prediction App")
st.markdown("""
This app predicts whether a loan applicant is a **Good** or **Bad** credit risk using the German Credit dataset.
It also provides feature insights and allows you to input new data for prediction.
""")

# Sidebar Navigation
menu = st.sidebar.selectbox("Select Activity", ["Data Overview", "Model Prediction", "Feature Insights"])

# Load dataset
df = load_data()

# Load model
model = load_model()

if menu == "Data Overview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔍 Data Summary")
    st.write(df.describe())

    st.subheader("💡 Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Risk', palette="Set2", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif menu == "Model Prediction":
    st.subheader("📥 Enter Applicant Details")

    age = st.slider("Age", 18, 75, 30)
    sex = st.selectbox("Sex", ['male', 'female'])
    job = st.selectbox("Job Type", [0, 1, 2, 3])
    housing = st.selectbox("Housing", ['own', 'rent', 'free'])
    saving_accounts = st.selectbox("Saving Accounts", ['little', 'moderate', 'quite rich', 'rich'])
    checking_account = st.number_input("Checking Account Balance (in DM)", 0, 20000, 100)
    credit_amount = st.number_input("Credit Amount (in DM)", 250, 20000, 1000)
    duration = st.slider("Loan Duration (in months)", 6, 72, 24)
    purpose = st.selectbox("Purpose", ['car', 'furniture/equipment', 'radio/TV', 'domestic appliances',
                                       'repairs', 'education', 'business', 'vacation/others'])

    input_dict = {
        'Age': age,
        'Sex': sex,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }

    input_df = pd.DataFrame([input_dict])

    # Load preprocessor
    preprocessor = joblib.load("preprocessor.pkl")
    input_processed = preprocessor.transform(input_df)

    if st.button("Predict Credit Risk"):
        prediction = model.predict(input_processed)[0]
        pred_proba = model.predict_proba(input_processed)[0]

        st.subheader("🔮 Prediction")
        st.write(f"**Credit Risk: {'Good' if prediction == 1 else 'Bad'}**")
        st.progress(pred_proba[1] if prediction == 1 else pred_proba[0])

elif menu == "Feature Insights":
    st.subheader("📈 Feature Importance")

    feature_names = joblib.load("feature_names.pkl")
    importances = model.feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    st.pyplot(fig)

    st.markdown("These features have the most influence on predicting credit risk. Use these insights to improve credit evaluation strategies.")

