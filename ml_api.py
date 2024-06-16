from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import numpy as np


app = FastAPI()

class model_input(BaseModel):

    High_Blood_Pressure: float
    Limited_Activities: float
    Use_of_Equipment: float
    Difficulty_Walking: float
    Arthritis_Diagnosis: float
    Doctor_Diagnosed_Arthritis: float
    Limited_in_Activities: float
    Quality_of_Life_Limited: float
    Pneumonia_Vaccine: float
    Internet_Usage: float
    Health_Coverage_Under_65: float
    Alcohol_Consumption: float
    Risk_Factor_BMI: float
    Weight_in_Kilograms: float
    Employment_Status: float
    Risk_Factors_for_Poor_Health: float
    Age_Group_5_Year_Intervals: float
    BMI_Categories: float
    Body_Mass_Index: float
    Risk_Factor_for_High_Blood_Pressure: float
    General_Health: float

with open('xgb_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):

    input_array = np.array([
        input_parameters.High_Blood_Pressure,
        input_parameters.Limited_Activities,
        input_parameters.Use_of_Equipment,
        input_parameters.Difficulty_Walking,
        input_parameters.Arthritis_Diagnosis,
        input_parameters.Doctor_Diagnosed_Arthritis,
        input_parameters.Limited_in_Activities,
        input_parameters.Quality_of_Life_Limited,
        input_parameters.Pneumonia_Vaccine,
        input_parameters.Internet_Usage,
        input_parameters.Health_Coverage_Under_65,
        input_parameters.Alcohol_Consumption,
        input_parameters.Risk_Factor_BMI,
        input_parameters.Weight_in_Kilograms,
        input_parameters.Employment_Status,
        input_parameters.Risk_Factors_for_Poor_Health,
        input_parameters.Age_Group_5_Year_Intervals,
        input_parameters.BMI_Categories,
        input_parameters.Body_Mass_Index,
        input_parameters.Risk_Factor_for_High_Blood_Pressure,
        input_parameters.General_Health
    ]).reshape(1, -1) 

    prediction = loaded_model.predict(input_array)

    if prediction[0] == 0.0:
        prediction_text = 'The person is not diabetic'
    elif prediction[0] == 1.0:
        prediction_text = 'The person is pre-diabetic'
    else:
        prediction_text = 'The person is diabetic'

    # Return JSON response with prediction
    return {'prediction': prediction_text}