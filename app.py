import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import sys
import subprocess

# Set page title and configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="centered"
)

# Page title and description
st.title("Diabetes Prediction Tool")
st.markdown("Enter patient information to predict diabetes risk")
st.markdown("---")

# Function to check if model exists and is compatible
def check_model():
    if not os.path.exists('diabetes_model.pkl'):
        return False, "Model file doesn't exist. Need to train model first."
    
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
            # Test a small prediction to ensure compatibility
            test_data = pd.DataFrame({
                'Age': [40],
                'Gender': ['M'],
                'BMI': [25.0],
                'SystolicBP': [120],
                'DiastolicBP': [80],
                'Smoker': ['No'],
                'PhysicalActivityLevel': ['Moderate'],
                'CholesterolLevel': ['Normal'],
                'FamilyHistory': ['No']
            })
            model.predict(test_data)
            return True, model
    except Exception as e:
        return False, str(e)

# Button to run the model script if needed
def run_model_script():
    st.info("Training the model... This might take a moment.")
    try:
        result = subprocess.run([sys.executable, 'diabetes_model.py'], 
                              capture_output=True, text=True, check=True)
        st.success("Model trained successfully! Refresh the page to use the prediction tool.")
        st.code(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error training model: {e}")
        st.code(e.stderr)

# Check if model is available and compatible
model_ok, model_or_error = check_model()

if not model_ok:
    st.error(f"Error with model: {model_or_error}")
    st.warning("Please train the model first by clicking the button below")
    if st.button("Train Model"):
        run_model_script()
else:
    model = model_or_error
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
        try:
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
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.warning("There might be compatibility issues with the model. Try retraining it.")
            if st.button("Retrain Model"):
                run_model_script() 