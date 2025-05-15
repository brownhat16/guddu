import requests
import streamlit as st

# Your backend API URL on Render
API_URL = "https://your-backend-app.onrender.com"

st.title("ML Model Demo")
st.write("Enter parameters to get predictions from the model")

# Create input fields for your model parameters
# For example, if your model predicts based on 4 parameters:
param1 = st.slider("Parameter 1", 0.0, 10.0, 5.0)
param2 = st.slider("Parameter 2", 0.0, 10.0, 5.0)
param3 = st.slider("Parameter 3", 0.0, 10.0, 5.0)
param4 = st.slider("Parameter 4", 0.0, 10.0, 5.0)

# Create a button to trigger the prediction
if st.button("Predict"):
    # Prepare the data to send to your API
    data = {
        "param1": param1,
        "param2": param2,
        "param3": param3,
        "param4": param4
    }

    # Make the API call to your backend
    response = requests.post(f"{API_URL}/predict", json=data)

    # Display the results
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Prediction: {prediction['result']}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
