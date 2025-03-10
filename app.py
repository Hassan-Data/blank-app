import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_rf_model.pkl")

# Define input fields for user
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict diabetes status:")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)
gender = st.selectbox("Gender", ["Male", "Female"])
polyuria = st.selectbox("Polyuria (Excessive Urination)", ["Yes", "No"])
polydipsia = st.selectbox("Polydipsia (Excessive Thirst)", ["Yes", "No"])
sudden_weight_loss = st.selectbox("Sudden Weight Loss", ["Yes", "No"])
weakness = st.selectbox("Weakness", ["Yes", "No"])
polyphagia = st.selectbox("Polyphagia (Excessive Hunger)", ["Yes", "No"])
genital_thrush = st.selectbox("Genital Thrush", ["Yes", "No"])
visual_blurring = st.selectbox("Visual Blurring", ["Yes", "No"])
itching = st.selectbox("Itching", ["Yes", "No"])
irritability = st.selectbox("Irritability", ["Yes", "No"])
delayed_healing = st.selectbox("Delayed Healing", ["Yes", "No"])
partial_paresis = st.selectbox("Partial Paresis", ["Yes", "No"])
muscle_stiffness = st.selectbox("Muscle Stiffness", ["Yes", "No"])
alopecia = st.selectbox("Alopecia (Hair Loss)", ["Yes", "No"])
obesity = st.selectbox("Obesity", ["Yes", "No"])

# Encoding user inputs to match model training
encoding_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
inputs = [
    age, encoding_map[gender], encoding_map[polyuria], encoding_map[polydipsia],
    encoding_map[sudden_weight_loss], encoding_map[weakness], encoding_map[polyphagia],
    encoding_map[genital_thrush], encoding_map[visual_blurring], encoding_map[itching],
    encoding_map[irritability], encoding_map[delayed_healing], encoding_map[partial_paresis],
    encoding_map[muscle_stiffness], encoding_map[alopecia], encoding_map[obesity]
]

# Predict button
tf st.button("Predict"):
    prediction = model.predict(np.array([inputs]))
    result = "Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes"
    st.write(f"Prediction: **{result}**")
