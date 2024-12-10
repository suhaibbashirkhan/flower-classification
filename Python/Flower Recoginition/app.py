import os
import numpy as np

import tensorflow as tf
import streamlit as st

# Set up the Streamlit header
st.header('Flower Classification CNN Model')

# List of flower names
flower_names = ['rose', 'sunflower', 'tulip']

# Load the pre-trained Keras model
model = tf.keras.models.load_model('Flower_Recog_Model.keras')

# Define the image classification function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = f"The image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%"
    return outcome

# File uploader for user to upload an image
uploaded_file = st.file_uploader('Upload an Image')

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_file_path = os.path.join('upload', uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, width=200, caption='Uploaded Image')

    # Classify the uploaded image and display the result
    classification_result = classify_images(temp_file_path)
    st.write(classification_result)