import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import sys
import subprocess
import plotly.graph_objects as go
import plotly.express as px
from copy import deepcopy

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

# Function to generate what-if scenarios
def generate_what_if_scenarios(base_input, model):
    scenarios = []
    
    # Current risk
    base_prob = model.predict_proba(base_input)[0][1]
    scenarios.append({"scenario": "Current", "risk": base_prob * 100, "change": 0})
    
    # BMI reduction scenarios
    if base_input['BMI'].values[0] > 23:
        bmi_inputs = []
        bmi_labels = []
        bmi_changes = []
        
        # Create multiple BMI reduction scenarios
        for reduction in [0.5, 1, 2, 3]:
            if base_input['BMI'].values[0] - reduction > 18.5:  # Ensure BMI doesn't go below healthy range
                modified_input = base_input.copy()
                modified_input['BMI'] = base_input['BMI'] - reduction
                bmi_inputs.append(modified_input)
                bmi_labels.append(f"BMI reduced by {reduction}")
                bmi_changes.append(-reduction)
        
        # Calculate probabilities for all BMI scenarios at once
        if bmi_inputs:
            combined_df = pd.concat(bmi_inputs, ignore_index=True)
            bmi_probs = model.predict_proba(combined_df)[:, 1]
            
            for i, (label, change) in enumerate(zip(bmi_labels, bmi_changes)):
                scenarios.append({
                    "scenario": label,
                    "risk": bmi_probs[i] * 100, 
                    "factor": "BMI",
                    "change": change
                })
    
    # Physical activity improvement
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    current_activity = base_input['PhysicalActivityLevel'].values[0]
    
    if current_activity == "Low":
        # Improve to moderate
        modified_input = base_input.copy()
        modified_input['PhysicalActivityLevel'] = "Moderate"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Increase activity to Moderate", 
            "risk": prob * 100,
            "factor": "Activity",
            "change": 1
        })
        
        # Improve to high
        modified_input = base_input.copy()
        modified_input['PhysicalActivityLevel'] = "High"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Increase activity to High", 
            "risk": prob * 100,
            "factor": "Activity",
            "change": 2
        })
    elif current_activity == "Moderate":
        # Improve to high
        modified_input = base_input.copy()
        modified_input['PhysicalActivityLevel'] = "High"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Increase activity to High", 
            "risk": prob * 100,
            "factor": "Activity",
            "change": 1
        })
    
    # Quit smoking
    if base_input['Smoker'].values[0] == "Yes":
        modified_input = base_input.copy()
        modified_input['Smoker'] = "No"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Quit smoking", 
            "risk": prob * 100,
            "factor": "Smoking",
            "change": -1
        })
    
    # Cholesterol improvement
    if base_input['CholesterolLevel'].values[0] == "High":
        modified_input = base_input.copy()
        modified_input['CholesterolLevel'] = "Borderline"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Improve cholesterol to Borderline", 
            "risk": prob * 100,
            "factor": "Cholesterol",
            "change": -1
        })
        
        modified_input = base_input.copy()
        modified_input['CholesterolLevel'] = "Normal"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Improve cholesterol to Normal", 
            "risk": prob * 100,
            "factor": "Cholesterol",
            "change": -2
        })
    elif base_input['CholesterolLevel'].values[0] == "Borderline":
        modified_input = base_input.copy()
        modified_input['CholesterolLevel'] = "Normal"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Improve cholesterol to Normal", 
            "risk": prob * 100,
            "factor": "Cholesterol",
            "change": -1
        })
    
    # Blood pressure improvement
    if base_input['SystolicBP'].values[0] > 120 or base_input['DiastolicBP'].values[0] > 80:
        modified_input = base_input.copy()
        
        # Create a moderate reduction
        systolic_reduction = min(10, base_input['SystolicBP'].values[0] - 120) if base_input['SystolicBP'].values[0] > 120 else 0
        diastolic_reduction = min(5, base_input['DiastolicBP'].values[0] - 80) if base_input['DiastolicBP'].values[0] > 80 else 0
        
        if systolic_reduction > 0 or diastolic_reduction > 0:
            modified_input['SystolicBP'] = base_input['SystolicBP'] - systolic_reduction
            modified_input['DiastolicBP'] = base_input['DiastolicBP'] - diastolic_reduction
            prob = model.predict_proba(modified_input)[0][1]
            scenarios.append({
                "scenario": f"Reduce BP by {systolic_reduction}/{diastolic_reduction} mmHg", 
                "risk": prob * 100,
                "factor": "Blood Pressure",
                "change": -(systolic_reduction + diastolic_reduction)/2
            })
    
    # Combined best case scenario (implement all positive changes)
    best_input = base_input.copy()
    changes_made = []
    
    # Reduce BMI to healthy (not below 18.5)
    if base_input['BMI'].values[0] > 25:
        target_bmi = max(18.5, min(base_input['BMI'].values[0] - 3, 25))
        best_input['BMI'] = target_bmi
        changes_made.append(f"BMI reduced to {target_bmi:.1f}")
    
    # Increase activity to High
    if base_input['PhysicalActivityLevel'].values[0] != "High":
        best_input['PhysicalActivityLevel'] = "High"
        changes_made.append("Activity increased to High")
    
    # Quit smoking
    if base_input['Smoker'].values[0] == "Yes":
        best_input['Smoker'] = "No"
        changes_made.append("Quit smoking")
    
    # Improve cholesterol to Normal
    if base_input['CholesterolLevel'].values[0] != "Normal":
        best_input['CholesterolLevel'] = "Normal"
        changes_made.append("Cholesterol improved to Normal")
    
    # Reduce blood pressure
    if base_input['SystolicBP'].values[0] > 120 or base_input['DiastolicBP'].values[0] > 80:
        best_input['SystolicBP'] = min(base_input['SystolicBP'].values[0], 120)
        best_input['DiastolicBP'] = min(base_input['DiastolicBP'].values[0], 80)
        changes_made.append("Blood pressure reduced to normal")
    
    # Calculate probability for best case
    if changes_made:
        prob = model.predict_proba(best_input)[0][1]
        scenarios.append({
            "scenario": "Optimal changes", 
            "risk": prob * 100,
            "factor": "Combined",
            "change": -5  # Symbolic value for sorting
        })
    
    return scenarios

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
                
            # Generate what-if scenarios
            st.markdown("---")
            st.subheader("Risk Projection Graph")
            st.markdown("This graph shows how changes to lifestyle factors could affect diabetes risk over time")
            
            scenarios = generate_what_if_scenarios(user_input, model)
            scenario_df = pd.DataFrame(scenarios)
            
            # Sort scenarios by risk
            scenario_df = scenario_df.sort_values(by="risk", ascending=False)
            
            # Create a visualization showing risk trajectories
            fig = go.Figure()
            
            # Add vertical line for diabetes threshold (50% risk)
            fig.add_shape(
                type="line",
                x0=50,
                y0=0,
                x1=50,
                y1=len(scenario_df) - 0.5,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add text annotation for threshold
            fig.add_annotation(
                x=50,
                y=len(scenario_df),
                text="Diabetes Risk Threshold",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=0
            )
            
            # Create the horizontal bar chart
            fig.add_trace(go.Bar(
                x=scenario_df["risk"],
                y=scenario_df["scenario"],
                orientation='h',
                marker=dict(
                    color=scenario_df["risk"],
                    colorscale='RdYlGn_r',
                    cmin=0,
                    cmax=100
                ),
                text=scenario_df["risk"].round(1).astype(str) + '%',
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title="Diabetes Risk by Scenario",
                xaxis_title="Risk Percentage (%)",
                yaxis_title="Scenario",
                height=400 + len(scenario_df) * 30,
                xaxis=dict(range=[0, 100]),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend analysis graph - show risk trajectory over time
            st.subheader("Risk Trajectory Over Time")
            st.markdown("This shows how risk might change over months with lifestyle modifications")
            
            # Create data for the trend lines
            months = list(range(0, 25, 3))  # 0, 3, 6, 9, 12, 15, 18, 21, 24 months
            
            # Base trend - no change
            base_risk = [probability * 100] * len(months)
            
            # Generate other improvement trends
            trends = {
                "No changes": base_risk,
                "Moderate improvements": [
                    max(probability * 100 * (1 - 0.1 * min(month/6, 1)), 10) 
                    for month in months
                ],
                "Significant lifestyle changes": [
                    max(probability * 100 * (1 - 0.25 * min(month/12, 1)), 5) 
                    for month in months
                ]
            }
            
            # Create the trend chart
            fig2 = go.Figure()
            
            # Add horizontal line for diabetes threshold
            fig2.add_shape(
                type="line",
                x0=0,
                y0=50,
                x1=24,
                y1=50,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Plot each trend line
            colors = {"No changes": "red", "Moderate improvements": "orange", "Significant lifestyle changes": "green"}
            
            for trend, values in trends.items():
                fig2.add_trace(go.Scatter(
                    x=months,
                    y=values,
                    mode='lines+markers',
                    name=trend,
                    line=dict(color=colors.get(trend, "blue"), width=3),
                    marker=dict(size=8)
                ))
            
            # Update layout
            fig2.update_layout(
                title="Diabetes Risk Trajectory Over Time",
                xaxis_title="Months",
                yaxis_title="Risk Percentage (%)",
                height=500,
                xaxis=dict(tickmode='array', tickvals=months),
                yaxis=dict(range=[0, 100]),
                legend=dict(y=0.99, x=0.01, bgcolor='rgba(255,255,255,0.8)'),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Disclaimer
            st.markdown("---")
            st.caption("Disclaimer: This is a simplified model for educational purposes only. The risk projections are estimates based on general trends, not personalized medical advice. Always consult healthcare professionals for medical guidance.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.warning("There might be compatibility issues with the model. Try retraining it.")
            if st.button("Retrain Model"):
                run_model_script() 