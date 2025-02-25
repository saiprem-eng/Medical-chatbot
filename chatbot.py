import streamlit as st
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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

description_list = {}
precautionDictionary = {}

# Load dictionaries
def load_dictionaries():
    global description_list, precautionDictionary
    description_list = {}
    precautionDictionary = {}

    try:
        with open('./symptom_Description.csv', mode='r') as csv_file:
            description_list = {row[0]: row[1] for row in csv.reader(csv_file)}
    except Exception as e:
        st.error(f"Error loading symptom descriptions: {e}")

    try:
        with open('./symptom_precaution.csv', mode='r') as csv_file:
            precautionDictionary = {row[0]: row[1:] for row in csv.reader(csv_file)}
    except Exception as e:
        st.error(f"Error loading symptom precautions: {e}")

load_dictionaries()

def predict_disease(symptoms):
    input_vector = np.zeros(len(cols))
    for symptom in symptoms:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1
    input_vector = input_vector.reshape(1, -1)
    return le.inverse_transform(clf.predict(input_vector))[0]

def get_precautions(disease):
    return precautionDictionary.get(disease, [])

def get_description(disease):
    return description_list.get(disease, 'No description available')

# Streamlit UI
st.set_page_config(page_title="HealthCare ChatBot", layout="centered")
st.markdown("""
<style>
body {
    background-color: #E8F5E9; /* Soft green background */
}

.stButton {
    background-color: #4CAF50; /* Green button */
    color: white; /* White text */
    border-radius: 8px; /* Rounded corners */
    height: 40px; /* Button height */
    font-size: 16px; /* Button font size */
}

.stButton:hover {
    background-color: #388E3C; /* Darker green on hover */
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 HealthCare ChatBot")
st.image("Healthcare-chatbot-hero-1024x780-1.webp", width=150)  # Add your medical-themed logo here

# Session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User name input
name = st.text_input("Enter your name:")

if name:
    with st.expander("Select Symptoms"):
        selected_symptoms = st.multiselect("Select your symptoms:", cols)
    
    days_sick = st.number_input("How many days have you been experiencing symptoms?", min_value=1, step=1)
    severity = st.slider("Rate the severity of your symptoms:", 1, 10, 5)

if st.button("Predict Disease"):
    if selected_symptoms:
        predicted_disease = predict_disease(selected_symptoms)
        description = get_description(predicted_disease)
        precautions = get_precautions(predicted_disease)

        st.session_state.predicted_disease = predicted_disease  # Store in session state
        st.subheader(f"{name}, you may have: **{predicted_disease}**")
        st.write(description)
        st.subheader("Treatments:")
        for i, precaution in enumerate(precautions, 1):
            if precaution:
                st.write(f"{i}. {precaution}")

        # Save chat history
        st.session_state.chat_history.append(f"{name} selected: {selected_symptoms} - Predicted: {predicted_disease}")
    else:
        st.warning("Please select at least one symptom.")
        
# Display chat history
if st.button("Show Chat History"):
    for message in st.session_state.chat_history:
        st.write(message)

# Option to download report
if 'predicted_disease' in st.session_state:  # Check if predicted_disease is stored in session state
    predicted_disease = st.session_state.predicted_disease
    report_data = f"User: {name}\nSymptoms: {selected_symptoms}\nPredicted Disease: {predicted_disease}\n"
    st.download_button("Download Report", report_data, "report.txt", mime="text/plain")
else:
    st.warning("Please predict the disease before downloading the report.")

# Sidebar for additional information
st.sidebar.header("Additional Information")
if st.sidebar.checkbox("Show FAQs"):
    st.sidebar.write("1. How does the chatbot work?")
    st.sidebar.write("2. What data is used for predictions?")
