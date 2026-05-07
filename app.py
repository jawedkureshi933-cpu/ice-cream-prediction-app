import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
#title
st.title("🍦 Ice Cream Sales Prediction App")

st.write("Pridict ice cream sales based on tempreature")

# input from user 
temp = st.number_input("Enter Temperature (°C)", value=25.0)

# Predict button

if st.button("Predict Sales"):
    input_data = np.array([[temp, temp**2]])
    prediction = model.predict(input_data)
    prediction_value = prediction[0][0]  # extract actual number
    st.success(f"Prediction Ice Cream Sales: {prediction_value:.2f} units")

#-------------------
#show Dataset + graph
#-------------------  

st.subheader("📊 Dataset Visualization")

df = pd.read_csv("IceCreamSalesData.csv")

fig, ax = plt.subplots()
ax.scatter(df['Temperature (°C)'], df['Ice Cream Sales (units)'])
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Sales")

st.pyplot(fig)

