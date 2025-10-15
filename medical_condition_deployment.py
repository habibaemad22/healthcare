
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.preprocessing import RobustScaler


# Page configuration
st.set_page_config(layout='wide', page_title='Medical Condition Prediction')

# Add custom CSS for dark theme
st.markdown(
    """
    <style>
    /* Body background */
    .stApp {
        background-color: #1e1e1e;  /* خلفية غامقة */
        color: #ffffff;              /* نص أبيض */
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title style */
    h1 {
        color: #ffffff;
        background-color: #333333;  /* خلفية أغمق للعنوان */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }

    /* Sidebar background */
    .css-1d391kg { 
        background-color: #2e2e2e;   /* خلفية غامقة للسايدبار */
        padding: 20px;
        border-radius: 10px;
        color: #ffffff;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
    }

    /* Hover effect for button */
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
    }

    /* Inputs style */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        font-size: 16px;
        color: #ffffff;                /* نص أبيض */
        background-color: #3a3a3a;     /* خلفية Inputs غامقة */
    }
    </style>
    """, unsafe_allow_html=True
)

# Title
html_title = """<h1> Medical Condition Prediction Project </h1>"""
st.markdown(html_title, unsafe_allow_html=True)

# Load cleaned data and trained model
df = pd.read_csv('cleaned_data.csv', index_col=0)
model = joblib.load('lr_model.pkl')  # replace with your trained model file

# Prepare LabelEncoder to decode predicted class
le = LabelEncoder()
y = df['Medical Condition']
le.fit(y)

# Columns used for prediction
input_cols = ['Age', 'Gender', 'Blood Type', 'Insurance Provider', 'Billing Amount',
              'Admission Type', 'Medication', 'Test Results', 'Length of Stay']

# Sidebar / Input widgets
age = st.sidebar.slider('Patient Age', int(df.Age.min()), int(df.Age.max()))
gender = st.sidebar.radio('Gender', df.Gender.unique())
blood_type = st.selectbox('Blood Type', df['Blood Type'].unique())
insurance = st.selectbox('Insurance Provider', df['Insurance Provider'].unique())
billing_amount = st.number_input('Billing Amount', float(df['Billing Amount'].min()), float(df['Billing Amount'].max()))
admission_type = st.selectbox('Admission Type', df['Admission Type'].unique())
medication = st.selectbox('Medication', df['Medication'].unique())
test_results = st.selectbox('Test Results', df['Test Results'].unique())
length_of_stay = st.number_input('Length of Stay (days)', int(df['Length of Stay'].min()), int(df['Length of Stay'].max()))

# Generate input DataFrame
new_data = pd.DataFrame([[age, gender, blood_type, insurance, billing_amount,
                          admission_type, medication, test_results, length_of_stay]],
                        columns=input_cols)

# Predict button
if st.button('Predict Medical Condition'):
    pred_encoded = model.predict(new_data)[0]               # Predict encoded label
    pred_label = le.inverse_transform([pred_encoded])[0]    # Decode to original label
    st.write(f'Predicted Medical Condition: **{pred_label}**')
