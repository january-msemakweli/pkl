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
import base64
from datetime import datetime
import time
from PIL import Image

# Set page title and configuration
st.set_page_config(
    page_title="Sukari Predictive Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load and apply custom CSS
with open('style.css', 'r') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Custom HTML components
def custom_header():
    # Load profile image
    try:
        profile_img_path = "Janu.jpg"
        if os.path.exists(profile_img_path):
            # Convert image to base64 for embedding
            with open(profile_img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            img_html = f'<img src="data:image/jpeg;base64,{img_data}" class="profile-image" alt="January G. Msemakweli">'
        else:
            img_html = ""
    except Exception as e:
        img_html = ""
        st.error(f"Error loading profile image: {str(e)}")
    
    header_html = f"""
    <div class="header-container">
        <div class="header-content">
            <h1 class="header-title">Sukari Predictive Model</h1>
            <p class="header-subtitle">Made by January G. Msemakweli</p>
            <div class="profile-container">{img_html}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def custom_card(title, content, key=None):
    card_html = f"""
    <div class="card" id="{key if key else ''}">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def custom_metric(label, value, suffix="", prefix=""):
    metric_html = f"""
    <div class="metric-container">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    return metric_html

def animated_text(text, tag="p", classname="fade-in"):
    return f'<{tag} class="{classname}">{text}</{tag}>'

def risk_gauge(risk_percent):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk", 'font': {'size': 22}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#1e5eff' if risk_percent < 30 else ('#ff9800' if risk_percent < 70 else '#f44336')},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 204, 150, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    
    return fig

# Tooltip helper
def tooltip(text, tooltip_text):
    return f'<span class="tooltip">{text}<span class="tooltiptext">{tooltip_text}</span></span>'

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
    progress_text = "Training the model... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Simulate training progress
        for percent_complete in range(0, 101, 10):
            time.sleep(0.1)  # Simulate work being done
            my_bar.progress(percent_complete, text=f"{progress_text} ({percent_complete}%)")
            
        result = subprocess.run([sys.executable, 'diabetes_model.py'], 
                              capture_output=True, text=True, check=True)
        my_bar.empty()
        
        st.success("✅ Model trained successfully! Refresh the page to use the prediction tool.")
        with st.expander("View training details"):
            st.code(result.stdout)
    except subprocess.CalledProcessError as e:
        my_bar.empty()
        st.error(f"❌ Error training model: {e}")
        with st.expander("View error details"):
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
        for reduction in [1, 3]:
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
        modified_input['CholesterolLevel'] = "Normal"
        prob = model.predict_proba(modified_input)[0][1]
        scenarios.append({
            "scenario": "Improve cholesterol to Normal", 
            "risk": prob * 100,
            "factor": "Cholesterol",
            "change": -2
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
            "scenario": "All optimal changes", 
            "risk": prob * 100,
            "factor": "Combined",
            "change": -5  # Symbolic value for sorting
        })
    
    return scenarios

# Apply the custom header
custom_header()

# Check if model is available and compatible
model_ok, model_or_error = check_model()

if not model_ok:
    st.markdown("""
    <div class="error-box">
        <h3>⚠️ Model Not Available</h3>
        <p>Error: {}</p>
        <p>Please train the model first by clicking the button below.</p>
    </div>
    """.format(model_or_error), unsafe_allow_html=True)
    
    if st.button("Train Model", key="train_model_button"):
        run_model_script()
else:
    model = model_or_error
    
    # Main layout - single page with clear sections
    st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
    
    # Patient Information form - optimized for mobile
    with st.form("prediction_form"):
        # Use more flexible column layout for better mobile compatibility
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", options=["M", "F"])
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            systolic_bp = st.number_input("Systolic BP", min_value=90, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
        
        with col2:
            smoker = st.selectbox("Smoker", options=["No", "Yes"])
            physical_activity = st.selectbox("Activity Level", options=["Low", "Moderate", "High"])
            cholesterol = st.selectbox("Cholesterol", options=["Normal", "Borderline", "High"])
            family_history = st.selectbox("Family History", options=["No", "Yes"])
        
        # Submit button - centered
        submit_button = st.form_submit_button(label="Calculate Risk")
    
    # Results section - only shown after prediction
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
            prediction = model.predict(user_input)[0]
            probability = model.predict_proba(user_input)[0][1]
            risk_percent = round(probability * 100, 1)
            scenarios = generate_what_if_scenarios(user_input, model)
            
            # Display section divider
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Assessment Results</div>', unsafe_allow_html=True)
            
            # Risk level
            if risk_percent >= 70:
                st.markdown('<div class="error-box"><h2>⚠️ High Risk</h2><p>This patient has a high risk of developing diabetes.</p></div>', unsafe_allow_html=True)
            elif risk_percent >= 30:
                st.markdown('<div class="warning-box"><h2>⚡ Moderate Risk</h2><p>This patient has a moderate risk of developing diabetes.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><h2>✅ Low Risk</h2><p>This patient has a low risk of developing diabetes.</p></div>', unsafe_allow_html=True)
            
            # Main result - now in 1 row for better mobile layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge
                st.plotly_chart(risk_gauge(risk_percent), use_container_width=True)
            
            with col2:
                # Probability as text
                st.markdown(f"<div style='text-align: center; margin: 20px 0;'><span style='font-weight: 500;'>Risk probability: </span><span style='font-size: 1.8rem; font-weight: 700; color: {'#f44336' if risk_percent >= 70 else '#ff9800' if risk_percent >= 30 else '#00cc88'};'>{risk_percent}%</span></div>", unsafe_allow_html=True)
            
                # Key risk factors heading
                st.markdown("<h4>Key Risk Factors</h4>", unsafe_allow_html=True)
                
                # Build risk factor tags
                risk_factors_html = "<div style='margin-top: 10px;'>"
                
                # BMI
                if bmi >= 30:
                    risk_factors_html += f"<div class='risk-tag high'>BMI: {bmi:.1f} (Obese)</div>"
                elif bmi >= 25:
                    risk_factors_html += f"<div class='risk-tag medium'>BMI: {bmi:.1f} (Overweight)</div>"
                else:
                    risk_factors_html += f"<div class='risk-tag low'>BMI: {bmi:.1f} (Healthy)</div>"
                
                # Blood pressure
                if systolic_bp >= 140 or diastolic_bp >= 90:
                    risk_factors_html += f"<div class='risk-tag high'>BP: {systolic_bp}/{diastolic_bp}</div>"
                elif systolic_bp >= 120 or diastolic_bp >= 80:
                    risk_factors_html += f"<div class='risk-tag medium'>BP: {systolic_bp}/{diastolic_bp}</div>"
                else:
                    risk_factors_html += f"<div class='risk-tag low'>BP: {systolic_bp}/{diastolic_bp}</div>"
                
                # Smoking
                if smoker == "Yes":
                    risk_factors_html += "<div class='risk-tag high'>Smoker</div>"
                
                # Activity
                if physical_activity == "Low":
                    risk_factors_html += "<div class='risk-tag high'>Low Activity</div>"
                elif physical_activity == "Moderate":
                    risk_factors_html += "<div class='risk-tag medium'>Moderate Activity</div>"
                
                # Cholesterol
                if cholesterol == "High":
                    risk_factors_html += "<div class='risk-tag high'>High Cholesterol</div>"
                elif cholesterol == "Borderline":
                    risk_factors_html += "<div class='risk-tag medium'>Borderline Cholesterol</div>"
                
                # Family History
                if family_history == "Yes":
                    risk_factors_html += "<div class='risk-tag high'>Family History</div>"
                
                risk_factors_html += "</div>"
                st.markdown(risk_factors_html, unsafe_allow_html=True)
            
            # Recommendations section
            st.subheader("Recommendations")
            recommendations = []
            
            if bmi >= 25:
                recommendations.append("🔻 Reduce BMI to healthy range (18.5-24.9)")
            
            if systolic_bp >= 120 or diastolic_bp >= 80:
                recommendations.append("💗 Lower blood pressure below 120/80")
            
            if smoker == "Yes":
                recommendations.append("🚭 Quit smoking")
            
            if physical_activity == "Low":
                recommendations.append("🏃 Increase physical activity level")
            
            if cholesterol == "High" or cholesterol == "Borderline":
                recommendations.append("🥗 Improve diet to reduce cholesterol")
            
            if not recommendations:
                recommendations.append("✅ Maintain current healthy lifestyle")
            
            for rec in recommendations:
                st.markdown(f"<div style='padding: 5px 0;'>{rec}</div>", unsafe_allow_html=True)
            
            # What-if scenarios section
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Risk Projection Graphs</div>', unsafe_allow_html=True)
            
            # Graphs - now stacked for better mobile view
            
            # First risk scenarios chart
            st.markdown("<h4>Impact of Lifestyle Changes</h4>", unsafe_allow_html=True)
            
            # Sort scenarios by risk for the chart
            scenario_df = pd.DataFrame(scenarios).sort_values(by="risk", ascending=False)
            
            # Create the horizontal bar chart for scenarios
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
                xaxis_title="Risk Percentage (%)",
                yaxis_title="Scenario",
                height=300,
                xaxis=dict(range=[0, 100]),
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Second risk trajectory chart
            st.markdown("<h4>Risk Trajectory Over Time</h4>", unsafe_allow_html=True)
            
            # Create data for the trend lines
            months = list(range(0, 25, 6))  # Fewer points for mobile: 0, 6, 12, 18, 24 months
            
            # Base trend - no change
            base_risk = [risk_percent] * len(months)
            
            # Generate improvement trends
            trends = {
                "No changes": base_risk,
                "Moderate improvements": [
                    max(risk_percent * (1 - 0.1 * min(month/6, 1)), 10) 
                    for month in months
                ],
                "Significant changes": [
                    max(risk_percent * (1 - 0.25 * min(month/12, 1)), 5) 
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
            colors = {"No changes": "#f44336", "Moderate improvements": "#ff9800", "Significant changes": "#00cc88"}
            
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
                xaxis_title="Months",
                yaxis_title="Risk Percentage (%)",
                height=300,
                xaxis=dict(tickmode='array', tickvals=months),
                yaxis=dict(range=[0, max(100, risk_percent * 1.1)]),
                legend=dict(orientation="h", y=1.1, x=0.0),
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10)
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.warning("There might be compatibility issues with the model. Try retraining it.")
            if st.button("Retrain Model"):
                run_model_script()
            
# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>© 2023 Sukari Predictive Model | This application is for educational purposes only</p>
    <p>Always consult healthcare professionals for medical advice</p>
</div>
""", unsafe_allow_html=True) 