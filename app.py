import streamlit as st
import pickle
import numpy as np

st.set_page_config("Heart Disease Detection")
st.title("Heart Disease Detection")
st.write("### This model is used to predict the heart disease")

def load_model():
    with open("prediction_model.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()
model = data["model"]

col1, col2 = st.columns(2)

with col1:
    Smoking = st.selectbox("Smoking", ("Yes", "No"))
    DiffWalking = st.selectbox("Do you have serious difficulty walking?", ("Yes", "No"))
    GenHealth = st.selectbox("General Health", ("Excellect", "Very Good", "Good", "Fair", "Poor"))
    SkinCancer = st.selectbox("Skin Cancer", ("Yes", "No"))
    PhysicalHealth = st.slider(label="Physical Health", step=5, max_value=50)

with col2:
    BMI = st.number_input(label="BMI", min_value=1, max_value=100)
    Stroke = st.selectbox("Stroke", ("Yes", "No"))
    Diabetic = st.selectbox("Diabetic", ("Yes", "No"))
    KidneyDisease = st.selectbox("Kidney Disease", ("Yes", "No"))
    Sex = st.selectbox("Gender", ("Male", "Female"))

button = st.button("Predict")

if button:
    Smoking = 1 if Smoking == "Yes" else 0
    DiffWalking = 1 if DiffWalking == "Yes" else 0
    SkinCancer = 1 if SkinCancer == "Yes" else 0
    Stroke = 1 if Stroke == "Yes" else 0
    Diabetic = 1 if Diabetic == "Yes" else 0
    KidneyDisease = 1 if KidneyDisease == "Yes" else 0

    if GenHealth == "Excellect":
        Health = 1
    elif GenHealth == "Very Good":
        Health = 2
    elif GenHealth == "Good":
        Health = 3
    elif GenHealth == "Fair":
        Health = 4
    else:
        Health = 5 

    Sex = 1 if Sex == "Male" else 0

    features = np.array([Smoking, DiffWalking,Health, SkinCancer, Stroke, Sex, Diabetic, KidneyDisease, BMI, PhysicalHealth]).reshape(1, -1)
    prediction = model.predict(features)

    if prediction == 1:
        output = "Postive"
        desc = 'This means there are chances that you can get a heart disease.'
    else:
        output = "Negative"
        desc = "This means that there are no chances to get a heart disease."

    st.subheader(f"The prediction is {output}\n")
    st.write(f"{desc}")