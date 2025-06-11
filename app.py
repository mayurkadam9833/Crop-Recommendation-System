# Importing necessary libraries
import streamlit as st 
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import pickle
import base64 

# Function to set background image using base64 encoding
def get_background(image_file): 
    with open(image_file,"rb")as file: 
        data=file.read()
        encoded = base64.b64encode(data).decode()

        css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Setting background image
get_background("farm.png")

# Loading pre-trained model
model=tf.keras.models.load_model("model.h5")

# Loading label encoder
with open("encoder.pkl","rb")as file: 
    label_encoder=pickle.load(file)

# Loading scaler for input normalization
with open("scaler.pkl","rb")as file: 
    scaler=pickle.load(file)


# Title of the Streamlit app
st.title("Crop Recommendation System")

# Short intro for the user
st.markdown('''🌾 Welcome to the Crop Recommendation System
This smart tool helps farmers and agricultural enthusiasts find the best crop to grow based on the current soil and weather conditions.

Just enter the values for:

🌱 Nitrogen (N) 🌿 Phosphorous (P) 🧪 Potassium (K) 🌡️ Temperature \n
💧 Humidity 🧬 pH Level ☔ Rainfall

Then click “Get Crop Recommendation” — and the system will use an advanced model to suggest the most suitable crop for your land.'''

)
# Sidebar section for user input
st.sidebar.header("🌿 Input Parameters")

# User inputs for each parameter
N=st.sidebar.number_input("Nitrogen",min_value=0.0,max_value=150.0)
P=st.sidebar.number_input("Phosphorous",min_value=0.0,max_value=150.0)
K=st.sidebar.number_input("Potassium",min_value=0.0,max_value=150.0)
temperature=st.sidebar.slider("Temperature (°C)",min_value=0.0,max_value=50.0)
humidity=st.sidebar.number_input("Humidity",min_value=0.0,max_value=100.0)
ph=st.sidebar.slider("pH",min_value=0.0,max_value=10.0)
rainfall=st.sidebar.number_input("Rainfall (mm)",min_value=0.0)

# When user clicks the button, run the prediction
if st.button("Get Crop Recommendation"): 
    # Create DataFrame from user inputs
    input_data=pd.DataFrame({
        "N":N, 
        "P":P, 
        "K":K, 
        "temperature":temperature, 
        "humidity":humidity, 
        "ph":ph, 
        "rainfall":rainfall
        },index=[0])
    
    # Scale the input data
    input_scale_data=scaler.transform(input_data)
    
    # Make prediction
    prediction=model.predict(input_scale_data) 

    # Get the index of the class with highest probability
    prediction_class=np.argmax(prediction,axis=1)[0]

     # Decode the prediction using the label encoder
    crop_recomdation=label_encoder.inverse_transform([prediction_class])[0]
    
     # Display the result
    st.success(f"According to condition your crop recommendation is {crop_recomdation}")
    st.balloons()


