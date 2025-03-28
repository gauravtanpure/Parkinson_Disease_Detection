import streamlit as st
from predict import predict_parkinson
from PIL import Image

# Streamlit app
st.title("Parkinson's Disease Detection")
st.write("Upload an image of a spiral or wave drawing to check for Parkinson's Disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Save the uploaded file temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make prediction
    result = predict_parkinson("temp_image.png")
    st.write(f"Prediction: **{result}**")