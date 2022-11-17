import streamlit as st
from tensorflow import keras
import numpy as np


model = keras.models.load_model('CNNmodel')
st.title("Covid X-ray Image Classifier!!")
uploaded_file = st.file_uploader("Choose a an image")
with st.container():
    if uploaded_file is not None:
        img = keras.utils.load_img(
        str(uploaded_file.name),
        grayscale=False,
        color_mode='rgb',
        target_size=(224, 224),
        interpolation='nearest',
        keep_aspect_ratio=False)

        st.image(img, caption="Patient's X-ray Image")

    if st.button('Predict'):
        prediction = model.predict(np.expand_dims(np.array(img),axis=0))
        if prediction[0][0] > 0.49:
            st.write('This patient has Covid')
        else:
            st.write('This patient does not have Covid')
