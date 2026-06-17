import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# 1. Download the model from HF Model Hub
# HF_TOKEN needs to be set as an environment variable in the Hugging Face Space
HF_TOKEN = os.environ.get("HF_TOKEN")

# Ensure it's in the environment for hf_hub_download (important for Streamlit Cloud/Spaces)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
else:
    # If running locally without environment var, try to get from userdata (Colab context)
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = HF_TOKEN
    except ImportError:
        print("HF_TOKEN not found in environment or Colab userdata.")

HF_USERNAME = "kthamaraikannan"  # Set your actual Hugging Face username here
MODEL_REPO  = f"{HF_USERNAME}/tourism-package-model"

# Debugging line: Print the HF_TOKEN status (first few chars) to the Streamlit logs
print(f"HF_TOKEN status (first 5 chars): {HF_TOKEN[:5] if HF_TOKEN else 'None/Empty'}")

@st.cache_resource
def load_model():
    # 2. Load the model
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

# 5. Prepare input data
input_data = pd.DataFrame([{
    "Age":                    age,
    "CityTier":               city_tier,
    "DurationOfPitch":        duration_of_pitch,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups":      num_followups,
    "PreferredPropertyStar":  preferred_star,
    "NumberOfTrips":          num_trips,
    "Passport":               1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar":                 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome":          monthly_income,
    "TypeofContact":          type_of_contact,
    "Occupation":             occupation,
    "Gender":                 gender,
    "ProductPitched":         product_pitched,
    "MaritalStatus":          marital_status,
    "Designation":            designation,
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
