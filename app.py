import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Paths for Model and Encoders
MODEL_PATH = "disease_prediction_model.pkl"
ENCODER_PATH = "label_encoders.pkl"
DISEASE_ENCODER_PATH = "disease_encoder.pkl"
DATASET_PATH = r"C:\Users\Shreya Saha\Desktop\ArogyaAI\arogyaai_dataset_expanded.csv"

# Load Model and Encoders
@st.cache_resource
def load_resources():
    """Loads the trained model and label encoders."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(DISEASE_ENCODER_PATH):
        st.error("❌ Model or encoders not found. Train the model first!")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        label_encoders = pickle.load(f)

    with open(DISEASE_ENCODER_PATH, "rb") as f:
        disease_encoder = pickle.load(f)

    return model, label_encoders, disease_encoder

# Load Dataset to get correct feature order
@st.cache_data
def load_dataset():
    """Loads the dataset to retrieve the correct feature order, removing unnamed columns."""
    if not os.path.exists(DATASET_PATH):
        st.error("❌ Dataset file not found!")
        return None
    
    df = pd.read_csv(DATASET_PATH)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    
    return df

# Load resources
model, label_encoders, disease_encoder = load_resources()
df = load_dataset()

# Ensure dataset was loaded successfully
if df is not None:
    expected_columns = df.columns.tolist()
    
    # Remove 'Patient ID' and 'Disease' since they are not features
    if "Patient ID" in expected_columns:
        expected_columns.remove("Patient ID")
    if "Disease" in expected_columns:
        expected_columns.remove("Disease")
else:
    expected_columns = []  # Fallback in case dataset is missing

# Streamlit UI
st.title("🩺 ArogyaAI - Smart Disease Prediction")

# **Navigation Tabs**
tab1, tab2 = st.tabs(["🔍 Predict Disease", "📊 View Dataset"])

# **Tab 1: Disease Prediction**
with tab1:
    st.subheader("📝 Enter Patient Details")

    # Define user inputs
    patient_id = st.text_input("Patient ID", placeholder="E.g., P12345")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms (comma separated)", placeholder="E.g., Fever, Cough, Fatigue")
    medical_history = st.text_area("Medical History", placeholder="E.g., Diabetes, Hypertension")
    region = st.text_input("Region", placeholder="E.g., Urban, Rural")
    language = st.selectbox("Language", ["English", "Hindi", "Other"])
    treatment_given = st.text_area("Treatment Given", placeholder="E.g., Paracetamol, Antibiotics")
    recovery_status = st.selectbox("Recovery Status", ["Recovered", "Not Recovered"])
    occupation = st.text_input("Occupation", placeholder="E.g., Teacher, Engineer")
    smoking_status = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
    vaccination_status = st.selectbox("Vaccination Status", ["Vaccinated", "Not Vaccinated"])
    chronic_conditions = st.text_area("Chronic Conditions", placeholder="E.g., Asthma, Heart Disease")
    hospitalization = st.selectbox("Hospitalization", ["Yes", "No"])
    days_to_recovery = st.number_input("Days to Recovery", min_value=0, max_value=365, step=1)
    duration = st.number_input("Duration (days)", min_value=0, max_value=365, step=1)
    severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])

    # Convert input data to DataFrame
    input_data = pd.DataFrame([[age, gender, symptoms, medical_history, region, language,
                                treatment_given, recovery_status, occupation, smoking_status, 
                                bmi, vaccination_status, chronic_conditions, hospitalization, 
                                days_to_recovery, duration, severity]],
                              columns=["Age", "Gender", "Symptoms", "Medical History", "Region",
                                       "Language", "Treatment Given", "Recovery Status", "Occupation",
                                       "Smoking Status", "BMI", "Vaccination Status", "Chronic Conditions",
                                       "Hospitalization", "Days to Recovery", "Duration", "Severity"])

    # **Fix: Ensure column order matches training dataset**
    missing_cols = [col for col in expected_columns if col not in input_data.columns]
    extra_cols = [col for col in input_data.columns if col not in expected_columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    if extra_cols:
        st.warning(f"⚠️ Ignoring unexpected extra columns: {extra_cols}")

    # Reorder columns to match training data
    input_data = input_data[expected_columns]

    # **Encode categorical values**
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype(str)

            # **Fix: Handle unseen categories dynamically**
            encoder = label_encoders.get(col)
            if encoder:
                if input_data[col][0] not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, input_data[col][0])
                input_data[col] = encoder.transform([input_data[col][0]])

    # **Prediction Button**
    if st.button("🔍 Predict Disease"):
        try:
            prediction_encoded = model.predict(input_data)
            predicted_disease = disease_encoder.inverse_transform(prediction_encoded)[0]

            st.success(f"✅ **Predicted Disease: {predicted_disease}**")
            st.write("📌 Ensure follow-up with a healthcare provider for accurate diagnosis.")

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# **Tab 2: View Dataset**
with tab2:
    st.subheader("📊 ArogyaAI Dataset")

    if df is not None:
        st.write("🔍 **Dataset Overview:**")
        st.dataframe(df.head())

        # **Show Dataset Summary**
        st.write("📌 **Dataset Summary**")
        st.write(df.describe())

        # **Show Missing Values**
        missing_values = df.isnull().sum()
        if missing_values.any():
            st.write("⚠️ **Missing Values in Dataset:**")
            st.write(missing_values[missing_values > 0])
        else:
            st.write("✅ No missing values in dataset.")

        # **Enable Search**
        search_col = st.selectbox("🔎 Select Column to Search", df.columns)
        search_val = st.text_input("🔍 Search Value")
        if search_val:
            filtered_df = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
            st.write(f"🔹 **Filtered Results for '{search_val}' in '{search_col}':**")
            st.dataframe(filtered_df)

# **Footer**
st.markdown("---")
st.write("© 2025 **ArogyaAI** | Developed with ❤️ by Shreya Saha")
