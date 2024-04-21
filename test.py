# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:12:15 2024

@author: aditya
"""
import os
import pickle
import streamlit as st
import pandas as pd
import base64
from streamlit_option_menu import option_menu

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/trainedmodel2.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/trainedmodel1.sav', 'rb'))
# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart disease Prediction'],
                           icons=['activity', 'heart'],
                           default_index=0)

# Custom CSS for the main container to set background image and style the typewriter effect with rainbow color
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,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");
        background-size: cover;
    }}
    @keyframes typing {{
        0% {{ width: 0; color: red }}
        10% {{ color: orange }}
        
        30% {{ color: green }}
        
        50% {{ color: indigo }}
        
        70% {{ color: pink }}
        80% {{ color: cyan }}
        90% {{ color: magenta }}
        100% {{ width: 100%; color: red }}
    }}
    .typing-animation {{
        overflow: hidden;
        animation: typing 7s steps(40, end) infinite alternate;
        white-space: nowrap;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Function to save input and output data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Function to download CSV file
def download_csv(data, filename):
    csv = pd.DataFrame(data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Diabetes prediction page
if selected == 'Diabetes Prediction':
    # Page title with typewriter effect and rainbow color
    st.markdown('<h1 class="typing-animation">Diabetes Prediction using ML</h1>', unsafe_allow_html=True)

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17)
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0, max_value=2)
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200)
        Insulin = st.number_input('Insulin Level', min_value=0, max_value=890)
        BMI = st.number_input('BMI value', min_value=0, max_value=70)
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=140)
        Age = st.number_input('Age between 20 to 100', min_value=20, max_value=100)

    # Code for Prediction
    diab_diagnosis = ''
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
        
        # Save input and output to CSV
        data = {'Pregnancies': [Pregnancies],
                'Glucose': [Glucose],
                'BloodPressure': [BloodPressure],
                'SkinThickness': [SkinThickness],
                'Insulin': [Insulin],
                'BMI': [BMI],
                'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
                'Age': [Age],
                'Diagnosis': [diab_diagnosis]}
        save_to_csv(data, 'diabetes_prediction_output.csv')
        
        # Download CSV button
        st.markdown(download_csv(data, 'diabetes_prediction_output.csv'), unsafe_allow_html=True)
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart disease Prediction':
    # Page title with typewriter effect and rainbow color
    st.markdown('<h1 class="typing-animation">Heart Disease Prediction using ML</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age between 1 to 100', min_value=1, max_value=100)
        trestbps = st.number_input('Resting Blood Pressure', min_value=90, max_value=200)
        restecg = st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2)
    with col2:
        sex = st.number_input('Sex 1 = male 0=female', min_value=0, max_value=1)
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=110, max_value=570)
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=60, max_value=210)
    with col3:
        cp = st.number_input('Chest Pain types', min_value=0, max_value=3)
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl', min_value=0, max_value=1)
        exang = st.number_input('Exercise Induced Angina', min_value=0, max_value=1)
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0, max_value=6)
        slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2)
    with col2:
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=4)
        thal = st.number_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect', min_value=0, max_value=3)

    # Code for Prediction
    heart_diagnosis = ''
    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_prediction = heartdisease_model.predict([user_input])
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        
        # Save input and output to CSV
        data = {'Age': [age],
                'Sex': [sex],
                'Chest Pain types': [cp],
                'Resting Blood Pressure': [trestbps],
                'Serum Cholestoral': [chol],
                'Fasting Blood Sugar': [fbs],
                'Resting Electrocardiographic results': [restecg],
                'Maximum Heart Rate achieved': [thalach],
                'Exercise Induced Angina': [exang],
                'ST depression induced by exercise': [oldpeak],
                'Slope of the peak exercise ST segment': [slope],
                'Major vessels colored by flourosopy': [ca],
                'Thal': [thal],
                'Diagnosis': [heart_diagnosis]}
        save_to_csv(data, 'heart_disease_prediction_output.csv')
        
        # Download CSV button
        st.markdown(download_csv(data, 'heart_disease_prediction_output.csv'), unsafe_allow_html=True)
    st.success(heart_diagnosis)


















