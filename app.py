import streamlit as st
import numpy as np
import pickle
import os

model_path = "diabetes_rf_model_fixed.pkl"

# Ensure model file exists before loading
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please check the file path.")
    st.stop()

# Load the model safely
try:
    with open("diabetes_rf_model_fixed.pkl", "rb") as f:
        model = pickle.load(f)

    if not hasattr(model, "predict"):
        raise ValueError("❌ Loaded model is NOT trained properly.")

    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Ensure model is properly loaded
if not hasattr(model, "predict"):
    st.error("Model is not properly loaded or trained. Please check the model file.")
    st.stop()

# Define function for prediction
def predict_diabetes(inputs):
    input_array = np.array(inputs).reshape(1, -1)
    
    # Debugging: Print input shape
    st.write(f"Input shape: {input_array.shape}")

    prediction = model.predict(input_array)
    return "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"

# Streamlit UI
st.title("🩺 Early Diabetes Prediction App")

# User input fields
st.sidebar.header("Enter Patient Details:")

age = st.sidebar.slider("Age", 16, 90, 30)  # Age slider
gender = st.sidebar.radio("Gender", ("Male", "Female"))
polyuria = st.sidebar.radio("Polyuria (Excessive Urination)", ("Yes", "No"))
polydipsia = st.sidebar.radio("Polydipsia (Excessive Thirst)", ("Yes", "No"))
sudden_weight_loss = st.sidebar.radio("Sudden Weight Loss", ("Yes", "No"))
weakness = st.sidebar.radio("Weakness", ("Yes", "No"))
polyphagia = st.sidebar.radio("Polyphagia (Excessive Hunger)", ("Yes", "No"))
genital_thrush = st.sidebar.radio("Genital Thrush", ("Yes", "No"))
visual_blurring = st.sidebar.radio("Visual Blurring", ("Yes", "No"))
itching = st.sidebar.radio("Itching", ("Yes", "No"))
irritability = st.sidebar.radio("Irritability", ("Yes", "No"))
delayed_healing = st.sidebar.radio("Delayed Healing", ("Yes", "No"))
partial_paresis = st.sidebar.radio("Partial Paresis", ("Yes", "No"))
muscle_stiffness = st.sidebar.radio("Muscle Stiffness", ("Yes", "No"))
alopecia = st.sidebar.radio("Alopecia (Hair Loss)", ("Yes", "No"))
obesity = st.sidebar.radio("Obesity", ("Yes", "No"))

# Convert categorical inputs into numerical format
gender = 1 if gender == "Male" else 0
polyuria = 1 if polyuria == "Yes" else 0
polydipsia = 1 if polydipsia == "Yes" else 0
sudden_weight_loss = 1 if sudden_weight_loss == "Yes" else 0
weakness = 1 if weakness == "Yes" else 0
polyphagia = 1 if polyphagia == "Yes" else 0
genital_thrush = 1 if genital_thrush == "Yes" else 0
visual_blurring = 1 if visual_blurring == "Yes" else 0
itching = 1 if itching == "Yes" else 0
irritability = 1 if irritability == "Yes" else 0
delayed_healing = 1 if delayed_healing == "Yes" else 0
partial_paresis = 1 if partial_paresis == "Yes" else 0
muscle_stiffness = 1 if muscle_stiffness == "Yes" else 0
alopecia = 1 if alopecia == "Yes" else 0
obesity = 1 if obesity == "Yes" else 0

# Convert inputs to a list
user_inputs = [age, gender, polyuria, polydipsia, sudden_weight_loss, weakness, polyphagia, 
               genital_thrush, visual_blurring, itching, irritability, delayed_healing, 
               partial_paresis, muscle_stiffness, alopecia, obesity]

# Prediction Button
if st.button("🔍 Predict Diabetes"):
    result = predict_diabetes(user_inputs)
    st.success(f"🩺 Prediction Result: {result}")
