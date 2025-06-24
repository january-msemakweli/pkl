import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# 1. Load and explore the dataset
print("Loading and exploring the dataset...")
data = pd.read_csv('data.csv')
print(data.head())
print("\nDataset shape:", data.shape)
print("\nDataset info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values check:")
print(data.isnull().sum())

# 2. Handle categorical variables
print("\nHandling categorical variables...")
# Identify categorical columns
categorical_cols = ['Gender', 'Smoker', 'PhysicalActivityLevel', 'CholesterolLevel', 'FamilyHistory']
numerical_cols = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP']

# 3. Convert DevelopedDiabetes to binary
print("\nConverting target to binary...")
data['DevelopedDiabetes'] = data['DevelopedDiabetes'].map({'Yes': 1, 'No': 0})

# Define features and target
X = data.drop(['PatientID', 'DevelopedDiabetes'], axis=1)
y = data['DevelopedDiabetes']

# Create preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough'
)

# 4. Split data into training and test sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a logistic regression model
print("\nTraining logistic regression model...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

model.fit(X_train, y_train)

# 6. Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Make predictions on test data
print("\nPredictions on test data:")
test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print(test_results)

# 8. Plot feature importance/coefficients
print("\nPlotting feature importance...")

# Apply the preprocessor to get the transformed feature matrix
X_transformed = model.named_steps['preprocessor'].transform(X_train)

# Get coefficients from the model
coefficients = model.named_steps['classifier'].coef_[0]

# Create a simpler feature importance plot without trying to map exact names
plt.figure(figsize=(10, 6))
plt.bar(range(len(coefficients)), np.abs(coefficients))
plt.title('Feature Importance for Diabetes Prediction')
plt.xlabel('Feature Index')
plt.ylabel('Absolute Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# 9. Save the model
print("\nSaving the model...")
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'diabetes_model.pkl'") 