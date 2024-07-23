

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('car.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app
st.title("Car Price Prediction")
st.header("Input Features")


# Sidebar inputs
year = st.slider("Year", min_value=2000, max_value=2022, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=6.0)
kms_driven = st.number_input("Kms Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
transmission = st.radio("Transmission", ['Manual', 'Automatic'])
owner = st.slider("Owner", min_value=0, max_value=3, value=0)
Car_Age = st.slider("Car Age", min_value=1, max_value=20, value=5)

# Create input DataFrame with correct feature names and order
input_data = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Owner': [owner],
    'Car_Ager': [Car_Age],
    'Fuel_Type': [fuel_type],
    'Seller_Type': [seller_type],
    'Transmission': [transmission]
})

# Map categorical variables to dummy variables
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure all required columns are present and in the correct order
expected_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner', 'Car_Age',
                    'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual']

# Reorder columns to match the expected order
input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# Make prediction
predicted_price = model.predict(input_data)

# Display prediction
st.write(f"Predicted Selling Price:   {predicted_price[0]:.2f} lakhs")
