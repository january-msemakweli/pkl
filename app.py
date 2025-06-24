import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page title and configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

# Page title and description
st.title("Diabetes Prediction Tool")
st.markdown("Enter patient information to predict diabetes risk")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    with open('diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Please run diabetes_model.py first to generate the model.")
    model_loaded = False

if model_loaded:
    # Create form for user inputs
    with st.form("prediction_form"):
        st.subheader("Patient Information")
        
        # Create two columns for form fields
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=40)
            gender = st.selectbox("Gender", options=["M", "F"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
        
        with col2:
            smoker = st.selectbox("Smoker", options=["Yes", "No"])
            physical_activity = st.selectbox("Physical Activity Level", options=["Low", "Moderate", "High"])
            cholesterol = st.selectbox("Cholesterol Level", options=["Normal", "Borderline", "High"])
            family_history = st.selectbox("Family History of Diabetes", options=["Yes", "No"])
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict Diabetes Risk")
    
    # Make prediction when form is submitted
    if submit_button:
        # Create a dataframe with the user input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'BMI': [bmi],
            'SystolicBP': [systolic_bp],
            'DiastolicBP': [diastolic_bp],
            'Smoker': [smoker],
            'PhysicalActivityLevel': [physical_activity],
            'CholesterolLevel': [cholesterol],
            'FamilyHistory': [family_history]
        })
        
        # Make prediction
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)[0][1]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Display prediction
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk**: This patient is predicted to develop diabetes.")
        else:
            st.success(f"✅ **Low Risk**: This patient is predicted to not develop diabetes.")
        
        # Display probability
        st.write(f"Probability of developing diabetes: {probability:.2%}")
        
        # Risk explanation based on input factors
        st.subheader("Risk Factor Analysis")
        risk_factors = []
        
        if bmi > 30:
            risk_factors.append("- High BMI (>30) indicates obesity, a significant risk factor")
        if systolic_bp > 140 or diastolic_bp > 90:
            risk_factors.append("- Elevated blood pressure can increase diabetes risk")
        if smoker == "Yes":
            risk_factors.append("- Smoking is associated with increased diabetes risk")
        if physical_activity == "Low":
            risk_factors.append("- Low physical activity increases diabetes risk")
        if cholesterol == "High":
            risk_factors.append("- High cholesterol is associated with increased diabetes risk")
        if family_history == "Yes":
            risk_factors.append("- Family history of diabetes increases risk")
        
        if risk_factors:
            st.markdown("\n".join(risk_factors))
        else:
            st.markdown("No significant risk factors identified from the input.")
        
        # Disclaimer
        st.markdown("---")
        st.caption("Disclaimer: This is a simplified model for educational purposes only. Always consult a healthcare professional for medical advice.") 