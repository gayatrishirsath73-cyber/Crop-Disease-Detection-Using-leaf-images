import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# FIX: define model path correctly
model_path = "model.h5"

model = tf.keras.models.load_model(model_path, compile=False)

classes = [
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy"
]