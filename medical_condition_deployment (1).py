
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
import plotly.express as px

# Page configuration
st.set_page_config(layout='wide', page_title='Medical Condition Prediction')

# Custom dark theme CSS
st.markdown("""
<style>
.stApp {background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', sans-serif;}
h1 {color: #ffffff; background-color: #333333; padding: 20px; border-radius: 10px; text-align: center;}
.css-1d391kg { background-color: #2e2e2e; padding: 20px; border-radius: 10px; color: #ffffff; }
div.stButton > button:first-child { background-color: #4CAF50; color: white; font-size: 18px; padding: 10px 20px; border-radius: 8px; border: none; }
div.stButton > button:hover { background-color: #45a049; color: white; }
.stSlider, .stNumberInput, .stSelectbox, .stRadio { font-size: 16px; color: #ffffff; background-color: #3a3a3a; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1> Medical Condition Prediction Project </h1>", unsafe_allow_html=True)

# Load data & model
df = pd.read_csv('cleaned_data.csv', index_col=0)
model = joblib.load('lr_model.pkl')

le = LabelEncoder()
y = df['Medical Condition']
le.fit(y)

# Columns for input
input_cols = ['Age', 'Gender', 'Blood Type', 'Insurance Provider', 'Billing Amount',
              'Admission Type', 'Medication', 'Test Results', 'Length of Stay']

# Tabs for Prediction & Analysis
tab1, tab2 = st.tabs(["Prediction", "Analysis"])

with tab1:
    st.subheader("Predict Medical Condition")
    # Sidebar / Input widgets
    age = st.slider('Patient Age', int(df.Age.min()), int(df.Age.max()))
    gender = st.radio('Gender', df.Gender.unique())
    blood_type = st.selectbox('Blood Type', df['Blood Type'].unique())
    insurance = st.selectbox('Insurance Provider', df['Insurance Provider'].unique())
    billing_amount = st.number_input('Billing Amount', float(df['Billing Amount'].min()), float(df['Billing Amount'].max()))
    admission_type = st.selectbox('Admission Type', df['Admission Type'].unique())
    medication = st.selectbox('Medication', df['Medication'].unique())
    test_results = st.selectbox('Test Results', df['Test Results'].unique())
    length_of_stay = st.number_input('Length of Stay (days)', int(df['Length of Stay'].min()), int(df['Length of Stay'].max()))

    new_data = pd.DataFrame([[age, gender, blood_type, insurance, billing_amount,
                              admission_type, medication, test_results, length_of_stay]],
                            columns=input_cols)

    if st.button('Predict Medical Condition'):
        pred_encoded = model.predict(new_data)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        st.write(f'Predicted Medical Condition: **{pred_label}**')

with tab2:
    st.subheader("Exploratory Analysis")

    # 1️⃣ Age and Gender distribution per Medical Condition
    fig1 = px.box(df, x='Medical Condition', y='Age', color='Gender',
                  title='Age and Gender Distribution per Medical Condition')
    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Length of Stay per Medical Condition and Admission Type
    fig2 = px.box(df, x='Medical Condition', y='Length of Stay', color='Admission Type',
                  title='Length of Stay per Medical Condition and Admission Type')
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Medication and Billing Amount per Medical Condition (Treemap)
    fig3 = px.treemap(df, path=['Medical Condition', 'Medication'], values='Billing Amount',
                      title='Medication and Billing Amount per Medical Condition')
    st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Age vs Medical Condition (with all points)
    fig_age = px.box(df, x='Medical Condition', y='Age', color='Medical Condition',
                     title='Age Distribution per Medical Condition', points='all')
    st.plotly_chart(fig_age, use_container_width=True)

    # 5️⃣ Length of Stay vs Medical Condition (with all points)
    fig_stay = px.box(df, x='Medical Condition', y='Length of Stay', color='Medical Condition',
                      title='Length of Stay per Medical Condition', points='all')
    st.plotly_chart(fig_stay, use_container_width=True)
