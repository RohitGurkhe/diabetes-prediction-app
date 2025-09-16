import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction Dashboard")

def user_input_features():
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 122, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin", 0, 846, 79)
    bmi = st.number_input("BMI", 0.0, 67.1, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 33)

    data = {'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
scaled_df = scaler.transform(input_df)
prediction = model.predict(scaled_df)
prediction_proba = model.predict_proba(scaled_df)

st.subheader("Prediction")
st.write("Diabetes Positive" if prediction[0]==1 else "Diabetes Negative")

st.subheader("Prediction Probability")
st.write(prediction_proba)
