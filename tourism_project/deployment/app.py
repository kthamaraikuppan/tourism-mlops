import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# --- MONKEY-PATCH FOR VERSION MISMATCH ---
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
# -----------------------------------------

# 1. Download the model from HF Model Hub
HF_TOKEN = os.environ.get("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN
HF_USERNAME = "kthamaraikannan"
MODEL_REPO  = f"{HF_USERNAME}/tourism-package-model"

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.joblib",
        repo_type="model",
        token=HF_TOKEN,
    )
    return joblib.load(path)

# 3. Streamlit UI
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🏖️",
    layout="centered",
)
st.title("🏖️ Visit with Us — Wellness Tourism Package Predictor")
st.write("Fill in the customer details below to predict purchase likelihood.")

model = load_model()

# 4. Collect user input
st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age                 = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier           = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch   = st.number_input("Duration of Pitch (minutes)", 0, 60, 15)
    num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
    num_followups       = st.number_input("Number of Followups", 0, 10, 3)
    preferred_star      = st.selectbox("Preferred Property Star", [3, 4, 5])
    num_trips           = st.number_input("Avg Trips per Year", 0, 20, 2)

with col2:
    passport            = st.selectbox("Has Passport?", ["Yes", "No"])
    pitch_satisfaction  = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car             = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children        = st.number_input("Children Visiting (below 5)", 0, 5, 0)
    monthly_income      = st.number_input("Monthly Income", 0, 1000000, 25000)
    type_of_contact     = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    occupation          = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])

gender          = st.selectbox("Gender", ["Male", "Female"])
marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# 5. Prepare input data with correct column ordering
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person_visiting,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups
}])

# 6. Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ Customer is LIKELY to purchase the Wellness Package  (Probability: {probability:.2%})")
    else:
        st.warning(f"❌ Customer is UNLIKELY to purchase the Wellness Package  (Probability: {probability:.2%})")
