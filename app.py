import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado
MODEL_PATH = "model/fruits_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.title("Clasificación de frutas con cámara 📸🍌🍎")

# Capturar imagen desde la cámara
img_file = st.camera_input("Toma una foto")

if img_file is not None:
    # Abrir y procesar la imagen
    image = Image.open(img_file)
    image = image.resize((224, 224))  # Ajustar al tamaño esperado por el modelo
    img_array = np.array(image) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

    # Predicción
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]

    # Mostrar resultados
    st.image(image, caption="Imagen capturada", use_column_width=True)
    st.write(f"**Predicción:** Clase {pred_class}")
