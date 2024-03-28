import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
st.set_page_config(layout="wide")
# Load the crop classes
classes = pd.read_excel("class.xlsx")

# Load the stacking classifier model
stacking_classifier = joblib.load('stacking_classifier_model.pkl')

# Load the scaling parameters
scaling_params = joblib.load('scaling_params.pkl')

# Create a new MinMaxScaler and set the parameters
scaler = MinMaxScaler()
scaler.data_min_ = scaling_params["min"]
scaler.data_max_ = scaling_params["max"]
scaler.min_ = scaling_params["min_"]
scaler.scale_ = scaling_params["scaler_"]

# Streamlit layout
st.markdown("<h1 style='text-align: center;'>Crop Recommendation</h1>", unsafe_allow_html=True)
st.image("im.jpg", use_column_width=True, width=400)
# Define the left and right columns
col1, col2 = st.columns(2)

with col1:
    

    N_input = st.text_input("Enter N value:")
    st.write("You entered:", N_input)

    P_input = st.text_input("Enter P value:")
    st.write("You entered:", P_input)

    K_input = st.text_input("Enter K value:")
    st.write("You entered:", K_input)

    temperature_input = st.text_input("Enter temperature input:")
    st.write("You entered:", temperature_input)

    humidity_input = st.text_input("Enter humidity input:")
    st.write("You entered:", humidity_input)

    ph_input = st.text_input("Enter pH value:")
    st.write("You entered:", ph_input)

    rainfall_input = st.text_input("Enter rainfall value:")
    st.write("You entered:", rainfall_input)

# Predict button
predict = st.button("Predict")
if predict:
    # Perform prediction using the model and display the result
    value = stacking_classifier.predict(scaler.transform([[N_input, P_input, K_input, temperature_input, humidity_input, ph_input, rainfall_input]]))[0]
    st.write(classes["class"][value:value+1])
    
with col2:
    components.iframe("https://www.vikatan.com/agriculture", height=800)
