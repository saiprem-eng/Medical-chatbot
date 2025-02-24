# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import csv
# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# # Load datasets
# training = pd.read_csv('./Training.csv')
# testing = pd.read_csv('./Testing.csv')
# cols = training.columns[:-1]
# x = training[cols]
# y = training['prognosis']
# le = preprocessing.LabelEncoder()
# le.fit(y)
# y = le.transform(y)

# # Split data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Train classifiers
# clf = DecisionTreeClassifier().fit(x_train, y_train)
# model = SVC().fit(x_train, y_train)

# description_list = {}
# precautionDictionary = {}
# severityDictionary = {}

# # Load dictionaries
# def load_dictionaries():
#     global description_list, precautionDictionary, severityDictionary
#     description_list = {}
#     precautionDictionary = {}
#     severityDictionary = {}
    
#     # Load symptom descriptions
#     try:
#         with open('./symptom_Description.csv', mode='r') as csv_file:
#             description_list = {row[0]: row[1] for row in csv.reader(csv_file)}
#     except FileNotFoundError:
#         print("Error: symptom_Description.csv not found.")
#     except Exception as e:
#         print(f"An error occurred while reading symptom_Description.csv: {e}")
    
#     # Load symptom precautions
#     try:
#         with open('./symptom_precaution.csv', mode='r') as csv_file:
#             precautionDictionary = {row[0]: row[1:] for row in csv.reader(csv_file)}
#     except FileNotFoundError:
#         print("Error: symptom_precaution.csv not found.")
#     except Exception as e:
#         print(f"An error occurred while reading symptom_precaution.csv: {e}")
    
#     # Load symptom severity
#     try:
#         with open('./Symptom_severity.csv', mode='r') as csv_file:
#             severityDictionary = {row[0]: int(row[1]) for row in csv.reader(csv_file)}
#     except FileNotFoundError:
#         print("Error: Symptom_severity.csv not found.")
#     except ValueError:
#         print("Error: Invalid data format in Symptom_severity.csv.")
#     except Exception as e:
#         print(f"An error occurred while reading Symptom_severity.csv: {e}")

# load_dictionaries()

# def predict_disease(symptoms):
#     input_vector = np.zeros(len(cols))
#     for symptom in symptoms:
#         if symptom in cols:  # Ensure symptom exists in dataset
#             input_vector[cols.get_loc(symptom)] = 1
#     input_vector = input_vector.reshape(1, -1)  # Ensure correct shape for prediction
#     return le.inverse_transform(clf.predict(input_vector))[0]

# def get_precautions(disease):
#     return precautionDictionary.get(disease, [])

# def get_description(disease):
#     return description_list.get(disease, 'No description available')

# # Streamlit UI
# st.title("HealthCare ChatBot")
# st.write("This chatbot uses general machine learning algorithms and predicts diseases based on symptoms users provide.")

# # Ask for name
# name = st.text_input("Enter your name:")

# # Ask for number of days
# if name:
#     # Ask for symptoms
#     selected_symptoms = st.multiselect("Select your symptoms:", cols)

#     #Ask for no of days sick
#     days_sick = st.number_input("How many days have you been experiencing symptoms?", min_value=1, step=1)
    
#     # Predict button
#     test_button = st.button("Predict Disease")
    
#     if test_button:
#         if selected_symptoms:
#             predicted_disease = predict_disease(selected_symptoms)
#             description = get_description(predicted_disease)
#             precautions = get_precautions(predicted_disease)
            
#             st.subheader(f"{name}, you may have: {predicted_disease}")
#             st.write(description)
            
#             st.subheader("Precautions:")
#             for i, precaution in enumerate(precautions, 1):
#                 if precaution:
#                     st.write(f"{i}. {precaution}")
#         else:
#             st.warning("Please select at least one symptom.")

import streamlit as st
import pandas as pd
import numpy as np
import re
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
    description_list = {}
    precautionDictionary = {}
    severityDictionary = {}

    # Load symptom descriptions
    try:
        with open('./symptom_Description.csv', mode='r') as csv_file:
            description_list = {row[0]: row[1] for row in csv.reader(csv_file)}
    except FileNotFoundError:
        print("Error: symptom_Description.csv not found.")
    except Exception as e:
        print(f"An error occurred while reading symptom_Description.csv: {e}")

    # Load symptom precautions
    try:
        with open('./symptom_precaution.csv', mode='r') as csv_file:
            precautionDictionary = {row[0]: row[1:] for row in csv.reader(csv_file)}
    except FileNotFoundError:
        print("Error: symptom_precaution.csv not found.")
    except Exception as e:
        print(f"An error occurred while reading symptom_precaution.csv: {e}")

    # Load symptom severity
    try:
        with open('./Symptom_severity.csv', mode='r') as csv_file:
            severityDictionary = {row[0]: int(row[1]) for row in csv.reader(csv_file)}
    except FileNotFoundError:
        print("Error: Symptom_severity.csv not found.")
    except ValueError:
        print("Error: Invalid data format in Symptom_severity.csv.")
    except Exception as e:
        print(f"An error occurred while reading Symptom_severity.csv: {e}")

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

def get_treatments(disease):
    treatments = {
        'Disease A': "Treatment A",
        'Disease B': "Treatment B",
        'Disease C': "Treatment C",
    }
    return treatments.get(disease, 'No treatment information available.')


# Streamlit UI
st.title("HealthCare ChatBot")
st.write("This chatbot uses general machine learning algorithms and predicts diseases based on symptoms users provide.")

# Ask for name
name = st.text_input("Enter your name:")

# Ask for number of days
if name:
    # Ask for symptoms
    selected_symptoms = st.multiselect("Select your symptoms:", cols)

    # Ask for no of days sick
    days_sick = st.number_input("How many days have you been experiencing symptoms?", min_value=1, step=1)

    # Severity rating
    severity = st.slider("Rate the severity of your symptoms:", 1, 10, 5)

    # Predict button
    test_button = st.button("Predict Disease")

    if test_button:
        with st.spinner("Predicting..."):
            # if selected_symptoms:
            #     predicted_disease = predict_disease(selected_symptoms)
            #     description = get_description(predicted_disease)
            #     precautions = get_precautions(predicted_disease)
            #     )

            #     st.success(f"{name}, you may have: {predicted_disease}")
            #     st.write(description)
            #     st.write(f"**Suggested Treatment:** {treatment_info}")

            #     st.subheader("Precautions:")
            #     for i, precaution in enumerate(precautions, 1):
            #         if precaution:
            #             st.write(f"{i}. {precaution}")

            #     # Optionally, you can plot disease prevalence (sample data for illustration)
            #     plot_disease_prevalence(selected_symptoms)

            # else:
            #     st.warning("Please select at least one symptom.")

             if selected_symptoms:
                    predicted_disease = predict_disease(selected_symptoms)
                    description = get_description(predicted_disease)
                    precautions = get_precautions(predicted_disease)
                    treatment_info = get_treatments(predicted_disease
                    
                    st.subheader(f"{name}, you may have: {predicted_disease}")
                    st.write(description)
                    
                    st.subheader("Precautions:")
                    for i, precaution in enumerate(precautions, 1):
                        if precaution:
                            st.write(f"{i}. {precaution}")
                    
                    # Call the plot_disease_prevalence function
                    plot_disease_prevalence(selected_symptoms)
             else:
                    st.warning("Please select at least one symptom.")


   
   

# Sidebar for additional information
st.sidebar.header("Additional Information")
if st.sidebar.checkbox("Show FAQs"):
    st.sidebar.write("1. How does the chatbot work?")
    st.sidebar.write("2. What data is used for predictions?")
