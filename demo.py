import streamlit as st
import pandas as pd
import numpy as np
import pyttsx3
import re
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load datasets
training = pd.read_csv('./Training.csv')
testing = pd.read_csv('./Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train classifiers
clf = DecisionTreeClassifier().fit(x_train, y_train)
model = SVC().fit(x_train, y_train)

description_list = {}
precautionDictionary = {}
severityDictionary = {}

# Load dictionaries
def load_dictionaries():
    global description_list, precautionDictionary, severityDictionary
    with open('./symptom_Description.csv') as csv_file:
        description_list = {row[0]: row[1] for row in csv.reader(csv_file)}
    with open('./symptom_precaution.csv') as csv_file:
        precautionDictionary = {row[0]: row[1:] for row in csv.reader(csv_file)}
    with open('./Symptom_severity.csv') as csv_file:
        severityDictionary = {row[0]: int(row[1]) for row in csv.reader(csv_file)}

load_dictionaries()

def predict_disease(symptoms):
    input_vector = np.zeros(len(cols))
    for symptom in symptoms:
        if symptom in cols:  # Ensure symptom exists in dataset
            input_vector[cols.get_loc(symptom)] = 1
    input_vector = input_vector.reshape(1, -1)  # Ensure correct shape for prediction
    return le.inverse_transform(clf.predict(input_vector))[0]


def get_precautions(disease):
    return precautionDictionary.get(disease, [])

def get_description(disease):
    return description_list.get(disease, 'No description available')

# Streamlit UI
st.title("HealthCare ChatBot")
st.write("Enter your symptoms to get a diagnosis.")

selected_symptoms = st.multiselect("Select your symptoms:", cols)

days_sick = st.number_input("How many days have you been experiencing symptoms?", min_value=1, step=1)

test_button = st.button("Predict Disease")
if test_button:
    if selected_symptoms:
        predicted_disease = predict_disease(selected_symptoms)
        description = get_description(predicted_disease)
        precautions = get_precautions(predicted_disease)

        st.subheader(f"You may have: {predicted_disease}")
        st.write(description)
        
        st.subheader("Precautions:")
        for i, precaution in enumerate(precautions, 1):
            st.write(f"{i}. {precaution}")
    else:
        st.warning("Please select at least one symptom.")
