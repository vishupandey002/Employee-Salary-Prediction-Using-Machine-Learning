import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family:Arial;
}

.main{
    background:#F4F7FC;
}

.title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#0F172A;
}

.subtitle{
    text-align:center;
    color:#64748B;
    font-size:20px;
    margin-bottom:30px;
}

div[data-testid="stButton"] button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:10px;
    height:55px;
    font-size:18px;
    border:none;
}

div[data-testid="stButton"] button:hover{
    background:#1D4ED8;
}

.result{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    padding:30px;
    border-radius:15px;
    text-align:center;
    color:white;
    margin-top:20px;
}

.metric{
    font-size:42px;
    font-weight:bold;
}

.small{
    color:#E2E8F0;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:60px;
}
left_btn, center_btn, right_btn = st.columns([1, 2, 1])

with center_btn:
    predict = st.button(
        "💰 Predict Salary",
        use_container_width=True
    )
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("salary_prediction_pipeline.pkl")

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("salary_data.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()
with st.sidebar:

    st.title("💼 Employee Salary Predictor")

    st.markdown("---")

    st.write("### Project")

    st.write("""
Predict employee salaries using a Machine Learning model built with Python, Scikit-learn and Streamlit.
""")

    st.markdown("---")

    st.write("### Model")

    st.success("Linear Regression")

    st.markdown("---")

    st.write("Developer")
    st.write("Vishu Pandey")  

# -------------------------------
# PAGE HEADER
# -------------------------------

st.markdown(
    "<h1 class='title'>💼 Employee Salary Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning Powered Salary Estimation</p>",
    unsafe_allow_html=True
)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Average Salary", f"₹ {df['salary'].mean():,.0f}")

with col3:
    st.metric("Highest Salary", f"₹ {df['salary'].max():,.0f}")

with col4:
    st.metric("Industries", df["industry"].nunique())
st.markdown("---")
# -------------------------------
# CREATE TWO COLUMNS
# -------------------------------

left, right = st.columns(2)

# -------------------------------
# LEFT COLUMN
# -------------------------------

with left:

    st.subheader("👤 Employee Information")

    job_title = st.selectbox(
        "Job Title",
        sorted(df["job_title"].dropna().unique())
    )

    experience_years = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=40,
        value=2
    )

    education_level = st.selectbox(
        "Education Level",
        sorted(df["education_level"].dropna().unique())
    )

    skills_count = st.slider(
        "Skills Count",
        int(df["skills_count"].min()),
        int(df["skills_count"].max()),
        5
    )

# -------------------------------
# RIGHT COLUMN
# -------------------------------

with right:

    st.subheader("🏢 Company Information")

    industry = st.selectbox(
        "Industry",
        sorted(df["industry"].dropna().unique())
    )

    company_size = st.selectbox(
        "Company Size",
        sorted(df["company_size"].dropna().unique())
    )

    location = st.selectbox(
        "Location",
        sorted(df["location"].dropna().unique())
    )

    remote_work = st.selectbox(
        "Remote Work",
        sorted(df["remote_work"].dropna().unique())
    )

    certifications = st.slider(
        "Certifications",
        int(df["certifications"].min()),
        int(df["certifications"].max()),
        1
    )

st.write("")

left_btn, center_btn, right_btn = st.columns([1, 2, 1])

with center_btn:
    predict = st.button(
        "💰 Predict Salary",
        use_container_width=True
    )
if predict:

    input_data = pd.DataFrame({
        "job_title":[job_title],
        "experience_years":[experience_years],
        "education_level":[education_level],
        "skills_count":[skills_count],
        "industry":[industry],
        "company_size":[company_size],
        "location":[location],
        "remote_work":[remote_work],
        "certifications":[certifications]
    })

    prediction = model.predict(input_data)[0]

    monthly_salary = prediction / 12

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result">

    <h2>🎉 Prediction Successful</h2>

    <div class="metric">
    ₹ {prediction:,.0f}
    </div>

    <div class="small">
    Estimated Annual Salary
    </div>

    <br>

    <h3>Monthly Salary</h3>

    <h2>₹ {monthly_salary:,.0f}</h2>

    </div>
    """, unsafe_allow_html=True)

st.markdown("## 📋 Prediction Summary")

summary = pd.DataFrame({
    "Feature":[
        "Job Title",
        "Experience",
        "Education",
        "Skills",
        "Industry",
        "Company Size",
        "Location",
        "Remote Work",
        "Certifications"
    ],
    "Value":[
        job_title,
        experience_years,
        education_level,
        skills_count,
        industry,
        company_size,
        location,
        remote_work,
        certifications
    ]
})
summary["Value"] = summary["Value"].astype(str)
st.dataframe(summary, use_container_width=True)
st.markdown("---")

import matplotlib.pyplot as plt

st.subheader("📊 Salary Distribution")

fig, ax = plt.subplots(figsize=(8,4))
ax.hist(df["salary"], bins=20)
ax.set_xlabel("Salary")
ax.set_ylabel("Employees")

st.pyplot(fig)
with st.expander("ℹ️ About the Model"):

    st.write("""
### Algorithm
Linear Regression

### Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

### Description
This application predicts employee salaries based on experience, education, skills, industry, company size, location, remote work status, and certifications.
""")

st.markdown("---")

st.markdown("""
<div class='footer'>
Developed with ❤️ using Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)