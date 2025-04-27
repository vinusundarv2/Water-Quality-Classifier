import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and scaler
@st.cache_resource
def load_model():
    with open('D:\Desktop\data analytics projects\water quality\water_quality_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('D:\Desktop\data analytics projects\water quality\scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()

# Create the Streamlit app
st.title('🚰 Water Quality Safety Predictor')
st.write('Enter the water quality parameters to check if the water is safe to drink')

# Create input fields for all features
with st.form("water_quality_form"):
    st.header("Water Quality Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        aluminium = st.number_input('Aluminium (mg/L)', min_value=0.0, value=1.65)
        ammonia = st.number_input('Ammonia (mg/L)', min_value=0.0, value=9.08)
        arsenic = st.number_input('Arsenic (mg/L)', min_value=0.0, value=0.04)
        barium = st.number_input('Barium (mg/L)', min_value=0.0, value=2.85)
        cadmium = st.number_input('Cadmium (mg/L)', min_value=0.0, value=0.007)
        chloramine = st.number_input('Chloramine (mg/L)', min_value=0.0, value=0.35)
        chromium = st.number_input('Chromium (mg/L)', min_value=0.0, value=0.83)
        copper = st.number_input('Copper (mg/L)', min_value=0.0, value=0.17)
        flouride = st.number_input('Flouride (mg/L)', min_value=0.0, value=0.05)
        bacteria = st.number_input('Bacteria (MPN/100mL)', min_value=0.0, value=0.2)
    
    with col2:
        viruses = st.number_input('Viruses (MPN/100mL)', min_value=0.0, value=0.0)
        lead = st.number_input('Lead (mg/L)', min_value=0.0, value=0.054)
        nitrates = st.number_input('Nitrates (mg/L)', min_value=0.0, value=16.08)
        nitrites = st.number_input('Nitrites (mg/L)', min_value=0.0, value=1.13)
        mercury = st.number_input('Mercury (mg/L)', min_value=0.0, value=0.007)
        perchlorate = st.number_input('Perchlorate (mg/L)', min_value=0.0, value=37.75)
        radium = st.number_input('Radium (pCi/L)', min_value=0.0, value=6.78)
        selenium = st.number_input('Selenium (mg/L)', min_value=0.0, value=0.08)
        silver = st.number_input('Silver (mg/L)', min_value=0.0, value=0.34)
        uranium = st.number_input('Uranium (mg/L)', min_value=0.0, value=0.02)
    
    submitted = st.form_submit_button("Check Water Safety")

if submitted:
    # Create a dataframe from the input values
    input_data = pd.DataFrame({
        'aluminium': [aluminium],
        'ammonia': [ammonia],
        'arsenic': [arsenic],
        'barium': [barium],
        'cadmium': [cadmium],
        'chloramine': [chloramine],
        'chromium': [chromium],
        'copper': [copper],
        'flouride': [flouride],
        'bacteria': [bacteria],
        'viruses': [viruses],
        'lead': [lead],
        'nitrates': [nitrates],
        'nitrites': [nitrites],
        'mercury': [mercury],
        'perchlorate': [perchlorate],
        'radium': [radium],
        'selenium': [selenium],
        'silver': [silver],
        'uranium': [uranium]
    })
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    # Display results
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.success("✅ The water is predicted to be SAFE for consumption")
    else:
        st.error("❌ The water is predicted to be UNSAFE for consumption")
    
    
    # Show feature importance (if available)
    try:
        st.subheader("Most Important Factors")
        feature_importance = pd.DataFrame({
            'Feature': input_data.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(feature_importance.set_index('Feature'))
    except:
        pass

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info(
    """
    This application predicts water safety using a Random Forest Classifier model and decision tree classifier
    \nEnter the water quality parameters and click 'Check Water Safety' to get a prediction.
    \nThe model was trained on water quality data with 20 different parameters.
    """
)