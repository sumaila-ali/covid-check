# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:21:23 2020

@author: Sam
"""

#%% Libraries
import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

#%%
gender_dict = {0:"Male", 1: "Female"}

dry_cough_dict = {0:"No", 1:"Yes"}

sore_throat_dict = {0:"No", 1:"Yes"}

weakness_dict = {0:"No", 1:"Yes"}

breathing_dict = {0:"No", 1:"Yes"}

drowsiness_dict = {0:"No", 1:"Yes"}

pain_chest_dict = {0:"No", 1:"Yes"}

travel_history_dict = {0:"No", 1:"Yes"}

diabetes_dict = {0:"No", 1:"Yes"}

heart_dis_dict = {0:"No", 1:"Yes"}

lung_dis_dict = {0:"No", 1:"Yes"}

stroke_immunity_dict = {0:"No", 1:"Yes"}

sypmtoms_pregressed_dict = {0:"No", 1:"Yes"}

high_blood_pres_dict = {0:"No", 1:"Yes"}

kidney_dis_dict = {0:"No", 1:"Yes"}

appetite_change_dict = {0:"No", 1:"Yes"}

loss_sense_smell_dict = {0:"No", 1:"Yes"}


#%%
st.title("Covid Preliminary Test")

age_var = st.text_input("Age", "0")
if age_var.isdigit() != True:
    st.warning("Enter an Integer value")
else: 
    age = age_var

gender = st.selectbox("Gender",  options = list(gender_dict.keys()),
             format_func = lambda x: gender_dict[x])

temp_var = st.text_input("Body Temperature in Fahrenheit", "0")
if temp_var.isdigit() != True:
    st.warning("Enter an Integer value")
else: 
    temperature = temp_var
    
Dry_cough = st.selectbox("Are you experiencing Dry Cough?",
                      options = list(dry_cough_dict.keys()), 
                    format_func = lambda x: dry_cough_dict[x])

sore_throat = st.selectbox("Are you experiencing Sore Throat?",
                      options = list(sore_throat_dict.keys()), 
                    format_func = lambda x: sore_throat_dict[x])

weakness = st.selectbox("Are you experiencing General Body Weakness?",
                      options = list(weakness_dict.keys()), 
                    format_func = lambda x: weakness_dict[x])

breathing  = st.selectbox("Are you experiencing Difficulty Breathing?",
                      options = list(breathing_dict.keys()), 
                    format_func = lambda x: breathing_dict[x])

drowsiness  = st.selectbox("Do you experience episodic Drowsiness?",
                      options = list(drowsiness_dict.keys()), 
                    format_func = lambda x: drowsiness_dict[x])

pain_chest  = st.selectbox("Do you experience pesistent Chest Pains?",
                      options = list(pain_chest_dict.keys()), 
                    format_func = lambda x: pain_chest_dict[x])

travel = st.selectbox("Have you traveled to a covid hotspot or Do you leave in an area clasified as a hotspot?",
                      options = list(travel_history_dict.keys()), 
                    format_func = lambda x: travel_history_dict[x])

diabetes  = st.selectbox("Are you diabetic?",
                      options = list(diabetes_dict.keys()), 
                    format_func = lambda x: diabetes_dict[x])

heart  = st.selectbox("Do you have a Heart Disease History?",
                      options = list(heart_dis_dict.keys()), 
                    format_func = lambda x: heart_dis_dict[x])

lung = st.selectbox("Do you have a Lung Disease History?",
                      options = list(lung_dis_dict.keys()), 
                    format_func = lambda x: lung_dis_dict[x])

stroke = st.selectbox("Do you have a Medical History with Strokes?",
                      options = list(stroke_immunity_dict.keys()), 
                    format_func = lambda x: stroke_immunity_dict[x])

blood_pressure = st.selectbox("Do you experience episodes of High Blood Pressure?",
                      options = list(high_blood_pres_dict.keys()), 
                    format_func = lambda x: high_blood_pres_dict[x])

kidney = st.selectbox("Do you have any form of Kidney Disease?",
                      options = list(kidney_dis_dict.keys()), 
                    format_func = lambda x: kidney_dis_dict[x])

appetite = st.selectbox("Have you lost your appetite over the last 7 days?",
                      options = list(appetite_change_dict.keys()), 
                    format_func = lambda x: appetite_change_dict[x])

smell = st.selectbox("Has any of the symptoms lasted more than a week?",
                      options = list(loss_sense_smell_dict.keys()), 
                    format_func = lambda x: loss_sense_smell_dict[x])

symptoms = st.selectbox("Have you lost your sense of smell over the past 7 days?",
                      options = list(sypmtoms_pregressed_dict.keys()), 
                    format_func = lambda x: sypmtoms_pregressed_dict[x])


#%%
var_name = ["age","gender", "body temperature", "Dry Cough", "sore throat",
            "weakness", "breathing problem", "drowsiness", "pain in chest",
            "travel history to infected countries", "diabetes", "heart disease",
            "lung disease", "stroke or reduced immunity", "symptoms progressed",
            "high blood pressue", "kidney disease", "change in appetide","Loss of sense of smell"]

new_df = [[age, gender, temperature, Dry_cough, sore_throat, weakness, breathing, 
           drowsiness, pain_chest, travel, diabetes, heart, lung, stroke, symptoms, 
           blood_pressure, kidney, appetite, smell]]

new_df = pd.DataFrame(new_df, columns = var_name)

data = pd.read_excel("COVID-19.xlsx")
data.drop(["Corona result", "Sno"], axis = 1, inplace = True)

features = pd.concat([new_df, data], axis = 0)

load_clf = pickle.load(open('covid.pkl', 'rb'))

new_data = features[:1]

if st.button("Process"):
    prediction = load_clf.decision_function(new_data)
    st.subheader('Prediction')
    max_decision = np.amax(prediction)
    location = np.where(np.amax(prediction))
#    st.write(prediction_proba)
#    st.subheader('Prediction')
    if int(location[0]) == 0:
        st.success("You do not have Covid-19 however, wear your mask and practice social distancing at all time")
    elif int(location[0]) == 1:
        st.info("You may not have Covid-19 however, you could self-isolate just to validate that assertion.")
        st.info("Also, wear your mask and practice social distancing if you have to go out.")
    elif int(location[0]) == 2:
        st.warning("You are showing symptoms that are heavily linked to Covid-19")
        st.warning("Kindly visit the nearest testing center to get tested.")
        st.warning("Please, remember to wear your nose mask and practice social distancing")
        
