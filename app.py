import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('model.h5')
st.title("Brain Tumor Segmentation")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image) / 255.0
    prediction = model.predict(img_array.reshape(1, 224, 224, 3))
    st.success(f"Prediction: {'Tumor' if prediction[0][0] > 0.5 else 'No Tumor'}")
