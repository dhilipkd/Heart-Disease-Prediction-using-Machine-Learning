import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "heart_model.pkl")
model = joblib.load(model_path)

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align: center; color: #d6336c;'>❤️ Heart Disease Risk Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter patient medical details to assess heart disease risk</p>",
    unsafe_allow_html=True
)

st.divider()

st.sidebar.header("Patient Information")
Age = st.sidebar.number_input("Age", 1, 120)
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Blood_Pressure = st.sidebar.number_input("Blood Pressure")
Cholesterol_Level = st.sidebar.number_input("Cholesterol Level")
BMI = st.sidebar.number_input("BMI")
Sleep_Hours = st.sidebar.number_input("Sleep Hours")

st.sidebar.divider()
st.sidebar.header("Lifestyle Factors")
Exercise_Habits = st.sidebar.selectbox("Exercise Habits", ["Low", "Moderate", "High"])
Smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
Alcohol_Consumption = st.sidebar.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
Stress_Level = st.sidebar.selectbox("Stress Level", ["Low", "Medium", "High"])

st.sidebar.divider()
st.sidebar.header("Medical History")
Family_Heart_Disease = st.sidebar.selectbox("Family Heart Disease", ["Yes", "No"])
Diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"])
Triglyceride_Level = st.sidebar.number_input("Triglyceride Level")
Fasting_Blood_Sugar = st.sidebar.number_input("Fasting Blood Sugar")


input_data = pd.DataFrame({
    "Age": [Age],
    "Gender": [Gender],
    "Blood Pressure": [Blood_Pressure],
    "Cholesterol Level": [Cholesterol_Level],
    "Exercise Habits": [Exercise_Habits],
    "Smoking": [Smoking],
    "Family Heart Disease": [Family_Heart_Disease],
    "Diabetes": [Diabetes],
    "BMI": [BMI],
    "Alcohol Consumption": [Alcohol_Consumption],
    "Stress Level": [Stress_Level],
    "Sleep Hours": [Sleep_Hours],
    "Triglyceride Level": [Triglyceride_Level],
    "Fasting Blood Sugar": [Fasting_Blood_Sugar]
})

st.divider()

if st.button(" Predict Risk", use_container_width=True):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader(" Prediction Result")

    if pred == 1:
        st.error(f" High Risk of Heart Disease\n\nRisk Probability: **{prob:.2f}**")
    else:
        st.success(f" Low Risk of Heart Disease\n\nRisk Probability: **{prob:.2f}**")
