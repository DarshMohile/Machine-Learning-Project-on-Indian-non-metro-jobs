import streamlit as st
import pandas as pd
import joblib
import json

# Load model and selected features
model = joblib.load("model.pkl")
with open("selected_features.json", "r") as f:
    selected_features = json.load(f)

st.title("Prediction of Approximate Salary using Random Forest Regression Predictor")
st.write("Enter the details below:")

user_input = {}

# Categorize features
def get_features_by_prefix(prefix):
    return [f for f in selected_features if f.startswith(prefix)]

city_features = get_features_by_prefix("City_")
industry_features = get_features_by_prefix("Industry_")
job_role_features = get_features_by_prefix("Job_Role_")
skill_set_features = get_features_by_prefix("Skill_Set_")
experience_features = get_features_by_prefix("Experience_Level_")
company_features = get_features_by_prefix("Company_Name_")
weekday_features = get_features_by_prefix("Weekday_")
season_features = get_features_by_prefix("Season_")

# Get remaining numerical features
handled = set(city_features + industry_features + job_role_features + skill_set_features +
              experience_features + company_features + weekday_features + season_features)
numeric_features = [f for f in selected_features if f not in handled]

# Selectionbox
def one_hot_selectbox(label, options, prefix):
    selected = st.selectbox(label, options)
    for opt in options:
        feature = f"{prefix}{opt}"
        user_input[feature] = 1.0 if opt == selected else 0.0

one_hot_selectbox("City", [c.split("_")[1] for c in city_features], "City_")
one_hot_selectbox("Industry", [i.split("_")[1] for i in industry_features], "Industry_")
one_hot_selectbox("Job Role", [j.split("_")[1] for j in job_role_features], "Job_Role_")
one_hot_selectbox("Experience Level", [e.split("_")[1] for e in experience_features], "Experience_Level_")
one_hot_selectbox("Company", [c.split("_")[1] for c in company_features], "Company_Name_")
one_hot_selectbox("Weekday", [w.split("_")[1] for w in weekday_features], "Weekday_")
one_hot_selectbox("Season", [s.split("_")[1] for s in season_features], "Season_")

# Skill set selection
all_skills = [s.replace("Skill_Set_", "") for s in skill_set_features]
selected_skills = st.multiselect("Select Skill Set", all_skills)

for skill in all_skills:
    feature = f"Skill_Set_{skill}"
    user_input[feature] = 1.0 if skill in selected_skills else 0.0

# Numeric values
st.write("Enter Numeric Values:")
for feature in numeric_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Prediction
input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Value: ₹{prediction[0]:,.2f}")
