import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("salary_prediction_pipeline.pkl")

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)
st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.main-title{
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#1E3A8A;
}

.sub-title{
    text-align:center;
    color:#64748B;
    font-size:18px;
    margin-bottom:30px;
}

.salary-card{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-top:20px;
    box-shadow:0px 8px 20px rgba(0,0,0,0.2);
}

.salary{
    font-size:42px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)
st.title("💼 Employee Salary Prediction")
st.write("Enter employee details below to predict the estimated salary.")

# User Inputs
job_title = st.selectbox(
    "Job Title",
    ["Data Scientist", "Software Engineer", "Data Analyst", "Project Manager", "HR Manager"]
)

experience_years = st.number_input(
    "Years of Experience",
    min_value=0,
    max_value=40,
    value=2
)

education_level = st.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

skills_count = st.slider(
    "Number of Skills",
    1,
    20,
    5
)

industry = st.selectbox(
    "Industry",
    ["IT", "Finance", "Healthcare", "Education", "Manufacturing"]
)

company_size = st.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

location = st.selectbox(
    "Location",
    ["Urban", "Suburban", "Rural"]
)

remote_work = st.selectbox(
    "Remote Work",
    ["Yes", "No"]
)

certifications = st.slider(
    "Certifications",
    0,
    10,
    1
)

# Prediction
if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "job_title": [job_title],
        "experience_years": [experience_years],
        "education_level": [education_level],
        "skills_count": [skills_count],
        "industry": [industry],
        "company_size": [company_size],
        "location": [location],
        "remote_work": [remote_work],
        "certifications": [certifications]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Salary: ₹ {prediction:,.2f}")