
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Page configuration
st.set_page_config(layout='wide', page_title='Admission Type Prediction')

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

st.markdown("<h1> Admission Type Prediction Project </h1>", unsafe_allow_html=True)

# Load data & model
df = pd.read_csv('cleaned_data.csv', index_col=0)
model = joblib.load('lr_model.pkl')

le = LabelEncoder()
y = df['Admission_Type']
le.fit(y)

# Columns for input
input_cols = ['Age', 'Gender', 'Blood_Type', 'Insurance_Provider', "Medical_Condition",
              'Billing_Amount', 'Medication', 'Test_Results', 'Length_of_Stay']

# Tabs for Prediction & Analysis
tab1, tab2 = st.tabs(["Prediction", "Analysis"])

with tab1:
    st.subheader("Predict Admission Type")
    # Input widgets
    age = st.slider('Patient Age', int(df.Age.min()), int(df.Age.max()))
    gender = st.radio('Gender', df.Gender.unique())
    blood_type = st.selectbox('Blood_Type', df['Blood_Type'].unique())
    insurance = st.selectbox('Insurance_Provider', df['Insurance_Provider'].unique())
    billing_amount = st.number_input('Billing_Amount', float(df['Billing_Amount'].min()), float(df['Billing_Amount'].max()))
    admission_type = st.selectbox('Admission_Type', df['Admission_Type'].unique())
    medication = st.selectbox('Medication', df['Medication'].unique())
    test_results = st.selectbox('Test_Results', df['Test_Results'].unique())
    length_of_stay = st.number_input('Length_of_Stay (days)', int(df['Length_of_Stay'].min()), int(df['Length_of_Stay'].max()))

    new_data = pd.DataFrame([[age, gender, blood_type, insurance, "Medical_Condition",
                              billing_amount, medication, test_results, length_of_stay]],
                            columns=input_cols)

    if st.button('Predict Admission Type'):
        pred_encoded = model.predict(new_data)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        st.write(f'Predicted Admission Type: **{pred_label}**')

with tab2:
    st.subheader("Exploratory Analysis")

    # 1️⃣ Number of Patients per Admission_Type
    count_df = df['Admission_Type'].value_counts().reset_index()
    count_df.columns = ['Admission_Type', 'Count']
    fig1 = px.bar(count_df,
                  x='Admission_Type', 
                  y='Count',
                  title='Number of Patients per Admission_Type')
    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Average Length_of_Stay per Admission_Type
    avg_stay = df.groupby('Admission_Type')['Length_of_Stay'].mean().reset_index()
    fig2 = px.bar(avg_stay, x='Admission_Type', y='Length_of_Stay',
                  title='Average Length_of_Stay per Admission_Type')
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Average Billing_Amount per Admission_Type
    avg_bill = df.groupby('Admission_Type')['Billing_Amount'].mean().reset_index()
    fig3 = px.bar(avg_bill, x='Admission_Type', y='Billing_Amount',
                  title='Average Billing_Amount per Admission_Type')
    st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Gender Distribution per Admission_Type
    gender_count = df.groupby(['Admission_Type','Gender']).size().reset_index(name='Count')
    fig4 = px.bar(gender_count, x='Admission_Type', y='Count', color='Gender', barmode='group',
                  title='Gender Distribution per Admission_Type')
    st.plotly_chart(fig4, use_container_width=True)
