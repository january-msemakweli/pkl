# Diabetes Prediction Application

This application predicts the likelihood of a patient developing diabetes based on demographic and health data.

## Project Structure

- `data.csv` - Dataset with patient records
- `diabetes_model.py` - Script to build and train the prediction model
- `app.py` - Streamlit web application for making predictions
- `requirements.txt` - Required Python packages
- `diabetes_model.pkl` - Trained model (generated after running the model script)
- `feature_importance.png` - Plot showing the importance of different features (generated after running the model script)

## Setup Instructions

1. **Install Requirements**

   ```
   pip install -r requirements.txt
   ```

2. **Build and Train the Model**

   ```
   python diabetes_model.py
   ```

3. **Run the Streamlit App**

   ```
   streamlit run app.py
   ```

## Using the Application

1. Fill in the patient information form with the required health data
2. Click the "Predict Diabetes Risk" button
3. View the prediction results and risk factor analysis

## Model Details

- The application uses a logistic regression model trained on patient health data
- The model considers factors like BMI, blood pressure, physical activity level, etc.
- Features are preprocessed with one-hot encoding for categorical variables

## Dataset

The dataset contains the following features:
- Patient demographics: Age, Gender
- Health metrics: BMI, Blood Pressure (Systolic & Diastolic)
- Lifestyle factors: Smoking status, Physical Activity Level
- Medical factors: Cholesterol Level, Family History of Diabetes
- Target variable: Whether the patient developed diabetes

## Disclaimer

This application is for educational purposes only. Always consult healthcare professionals for medical advice. 