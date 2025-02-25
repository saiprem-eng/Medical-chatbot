# import streamlit as st
# import pandas as pd
# import numpy as np
# import csv
# from sklearn import preprocessing
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

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

# description_list = {}
# precautionDictionary = {}

# # Load dictionaries
# def load_dictionaries():
#     global description_list, precautionDictionary
#     description_list = {}
#     precautionDictionary = {}

#     try:
#         with open('./symptom_Description.csv', mode='r') as csv_file:
#             description_list = {row[0]: row[1] for row in csv.reader(csv_file)}
#     except Exception as e:
#         st.error(f"Error loading symptom descriptions: {e}")

#     try:
#         with open('./symptom_precaution.csv', mode='r') as csv_file:
#             precautionDictionary = {row[0]: row[1:] for row in csv.reader(csv_file)}
#     except Exception as e:
#         st.error(f"Error loading symptom precautions: {e}")

# load_dictionaries()

# def predict_disease(symptoms):
#     input_vector = np.zeros(len(cols))
#     for symptom in symptoms:
#         if symptom in cols:
#             input_vector[cols.get_loc(symptom)] = 1
#     input_vector = input_vector.reshape(1, -1)
#     return le.inverse_transform(clf.predict(input_vector))[0]

# def get_precautions(disease):
#     return precautionDictionary.get(disease, [])

# def get_description(disease):
#     return description_list.get(disease, 'No description available')

# # Streamlit UI
# st.set_page_config(page_title="HealthCare ChatBot", layout="centered")
# st.markdown("""
# <style>
# body {
#     background-color: #E8F5E9; /* Soft green background */
# }

# .stButton {
#     background-color: #4CAF50; /* Green button */
#     color: white; /* White text */
#     border-radius: 8px; /* Rounded corners */
#     height: 40px; /* Button height */
#     font-size: 16px; /* Button font size */
# }

# .stButton:hover {
#     background-color: #388E3C; /* Darker green on hover */
# }
# </style>
# """, unsafe_allow_html=True)

# st.title("🩺 HealthCare ChatBot")
# st.image("Healthcare-chatbot-hero-1024x780-1.webp", width=150)  # Add your medical-themed logo here

# # Session state for chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # User name input
# name = st.text_input("Enter your name:")

# if name:
#     with st.expander("Select Symptoms"):
#         selected_symptoms = st.multiselect("Select your symptoms:", cols)
    
#     days_sick = st.number_input("How many days have you been experiencing symptoms?", min_value=1, step=1)
#     severity = st.slider("Rate the severity of your symptoms:", 1, 10, 5)

# if st.button("Predict Disease"):
#     if selected_symptoms:
#         predicted_disease = predict_disease(selected_symptoms)
#         description = get_description(predicted_disease)
#         precautions = get_precautions(predicted_disease)

#         st.session_state.predicted_disease = predicted_disease  # Store in session state
#         st.subheader(f"{name}, you may have: **{predicted_disease}**")
#         st.write(description)
#         st.subheader("Treatments:")
#         for i, precaution in enumerate(precautions, 1):
#             if precaution:
#                 st.write(f"{i}. {precaution}")

#         # Save chat history
#         st.session_state.chat_history.append(f"{name} selected: {selected_symptoms} - Predicted: {predicted_disease}")
#     else:
#         st.warning("Please select at least one symptom.")
        
# # Display chat history
# if st.button("Show Chat History"):
#     for message in st.session_state.chat_history:
#         st.write(message)

# # Option to download report
# if 'predicted_disease' in st.session_state:  # Check if predicted_disease is stored in session state
#     predicted_disease = st.session_state.predicted_disease
#     report_data = f"User: {name}\nSymptoms: {selected_symptoms}\nPredicted Disease: {predicted_disease}\n"
#     st.download_button("Download Report", report_data, "report.txt", mime="text/plain")
# else:
#     st.warning("Please predict the disease before downloading the report.")

# # Sidebar for additional information
# st.sidebar.header("Additional Information")
# if st.sidebar.checkbox("Show FAQs"):
#     st.sidebar.write("1. How does the chatbot work?")
#     st.sidebar.write("2. What data is used for predictions?")


import streamlit as st

st.markdown("""
<style>
body {
    background-color: #F8F9FA; /* Very light gray background */
    font-family: 'Arial', sans-serif; /* Modern font */
}

.main .block-container {
    max-width: 900px; /* Adjust content width */
    padding-top: 3rem;
    padding-bottom: 3rem;
}

.stButton {
    background-color: #28A745; /* Calming green */
    color: white;
    border-radius: 8px;
    padding: 10px 20px; /* Adjust padding */
    font-size: 16px;
    border: none; /* Remove default border */
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    transition: background-color 0.3s ease; /* Smooth transition */
}

.stButton:hover {
    background-color: #218838; /* Darker green on hover */
    box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.15); /* More prominent shadow on hover */
}

.stTextInput, .stTextArea {
    border: 1px solid #CED4DA; /* Light gray border for input fields */
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
    box-shadow: inset 1px 1px 3px rgba(0, 0, 0, 0.05); /* Inner shadow */
}

.stTextInput:focus, .stTextArea:focus {
    border-color: #80BDFF; /* Blue border on focus */
    outline: none;
    box-shadow: inset 1px 1px 5px rgba(0, 0, 0, 0.1), 0 0 5px rgba(128, 189, 255, 0.2); /* Enhanced focus effect */
}

/* Chat bubble styling */
.chat-message {
    background-color: #FFFFFF; /* White background for messages */
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    clear: both; /* Ensure bubbles are displayed correctly */
}

.user-message {
    float: right; /* Align user messages to the right */
    background-color: #DCF8C6; /* Light green for user messages */
}

.bot-message {
    float: left; /* Align bot messages to the left */
}

/* Avatar styling (optional) */
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
}

/* Scrollbar styling (optional - customize as needed) */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Add any other custom styles here */

</style>
""", unsafe_allow_html=True)


# Example usage (replace with your actual chatbot logic)
st.title("Healthcare Chatbot")

# Example chat messages
st.markdown('<div class="chat-message bot-message"><img class="avatar" src="bot_avatar.png" alt="Bot Avatar"> Hello! How can I help you?</div>', unsafe_allow_html=True)  # Replace "bot_avatar.png"
st.markdown('<div class="chat-message user-message">I have a headache.</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-message bot-message"><img class="avatar" src="bot_avatar.png" alt="Bot Avatar">  I see.  Could you describe the pain?</div>', unsafe_allow_html=True)

user_input = st.text_input("Enter your message:")

if st.button("Send"):
    if user_input:
        st.markdown(f'<div class="chat-message user-message">{user_input}</div>', unsafe_allow_html=True)
        # Process user input and generate bot response here
        bot_response = "I am a bot. I cannot provide medical advice." # Replace with actual bot logic
        st.markdown(f'<div class="chat-message bot-message"><img class="avatar" src="bot_avatar.png" alt="Bot Avatar"> {bot_response}</div>', unsafe_allow_html=True)
