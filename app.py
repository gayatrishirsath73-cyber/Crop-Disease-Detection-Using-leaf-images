import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model

import tensorflow as tf

model = tf.keras.models.load_model("model.h5", compile=False)

# Classes (ONLY 2)
classes = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy"
]

st.title("Crop Disease Detection")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    img = image.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    result = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")