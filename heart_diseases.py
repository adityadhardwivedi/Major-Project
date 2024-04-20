# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
# loaded the saved model
loaded_model = pickle.load(open('E:/multiple-disease-prediction-streamlit-app-main/trainedmodel1.sav','rb'))

input_data = (62,0,1,138,294,1,1,106,0,1.9,2,3,6)
# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the numpy array as we are only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('person has not heart disease')
else:
  print('person has heart disease')