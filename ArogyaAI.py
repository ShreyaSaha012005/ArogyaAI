import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Set dataset path
file_path = r"C:\Users\Shreya Saha\Desktop\ArogyaAI\arogyaai_dataset_expanded.csv"

# Check if dataset exists
if not os.path.exists(file_path):
    print("❌ Dataset not found. Ensure the file is available at:", file_path)
    exit()

# Load dataset with error handling
try:
    df = pd.read_csv(file_path, encoding="utf-8")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# Display dataset info before processing
print("\n📌 Initial Dataset Info:")
print(df.info())

# Identify categorical and numerical columns
categorical_columns = [
    "Gender", "Symptoms", "Medical History", "Region",
    "Language", "Treatment Given", "Recovery Status",
    "Occupation", "Smoking Status", "Vaccination Status",
    "Chronic Conditions", "Hospitalization", "Severity"
]

numerical_columns = ["Age", "Duration", "BMI", "Days to Recovery"]

# Ensure numerical columns exist in the dataset
numerical_columns = [col for col in numerical_columns if col in df.columns]

# Convert numerical columns properly
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric values to NaN

# Fill missing values in numerical columns with column mean
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Drop Patient ID if present
if "Patient ID" in df.columns:
    df.drop(columns=["Patient ID"], inplace=True)

label_encoders = {}

# Ensure categorical columns exist in the dataset
categorical_columns = [col for col in categorical_columns if col in df.columns]

# Encode categorical columns
for col in categorical_columns:
    df[col] = df[col].astype(str)  # Ensure all categorical values are strings
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Define features (X) and target variable (y)
if "Disease" not in df.columns:
    print("❌ Column 'Disease' not found in dataset.")
    exit()

X = df.drop(columns=["Disease"])  # Exclude target variable
y = df["Disease"]

# Encode the target variable
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y)

# Save disease label encoder
with open("disease_encoder.pkl", "wb") as f:
    pickle.dump(disease_encoder, f)

# Check for rare diseases to prevent train_test_split error
disease_counts = df["Disease"].value_counts()
rare_diseases = disease_counts[disease_counts <= 1].index

if len(rare_diseases) > 0:
    print(f"⚠️ Warning: {len(rare_diseases)} diseases have only 1 occurrence. These will be removed to prevent errors.")
    df = df[~df["Disease"].isin(rare_diseases)]  # Remove rare diseases

# Re-define features (X) and target variable (y) after filtering rare diseases
X = df.drop(columns=["Disease"])
y = df["Disease"]

# Re-encode the target variable
y = disease_encoder.fit_transform(y)

# Save updated disease encoder
with open("disease_encoder.pkl", "wb") as f:
    pickle.dump(disease_encoder, f)

# Split dataset safely
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print("⚠️ Stratified split failed. Using random split instead.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with error handling
try:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
except Exception as e:
    print("❌ Error training model:", e)
    exit()

# Save trained model
with open("disease_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved successfully!")
