#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:04:33 2024

@author: vaishnaviawadhiya
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle. load(open('/Users/vaishnaviawadhiya/minor project/trained model. sav', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray (input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape (1, -1)
    prediction = loaded_model.predict (input_data_reshaped)
    print (prediction)
    if (prediction [0] == 0):
        print("hi i am not")
        return 'The person is not diabetic'
    else:
        print("hi i am")
        return 'The person is diabetic'

def main():
    st.set_page_config(
        page_title="Diabetes Prediction Web App",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    #giving a title
    st.title('Syndromic Surveillance')
    
    #getting the input data from the user
  
    Pregnancies = st.text_input( 'Number of Pregnancies')
    Glucose = st.text_input( 'Glucose level')
    BloodPressure = st.text_input ( 'BLood Pressure value')
    SkinThickness = st.text_input( 'Skin Thickness value')
    Insulin = st.text_input( 'Insulin Level')
    BMI = st.text_input ('BMI vaLue')
    DiabetesPedigreeFunction = st.text_input( 'Diabetes Pedigree Function value')
    Age = st.text_input( 'Age of the Person')
    
    #code for prediction
    diagnosis = ""

    #creating a button for prediction

    if st.button('Diabetes Test Result'):
        try:
            # Validate input data
            input_data = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), int(Age)

            ]

            diagnosis = diabetes_prediction(input_data)
            st.success(diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")

    
    
if __name__ == '__main__':
    main()
    
    
    
    