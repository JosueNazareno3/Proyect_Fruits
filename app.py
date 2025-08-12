import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar modelo
MODEL_PATH = "model/fruits_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de frutas en el mismo orden que tu entrenamiento
class_names = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

st.title("Clasificador de Frutas 🍎🍌🍇")
st.write("Toma una foto con tu cámara y detectaremos la fruta.")

# Abrir cámara
img_file = st.camera_input("Toma una foto")

if img_file is not None:
    # Leer imagen desde la cámara
    image = Image.open(img_file).convert('RGB')
    st.image(image, caption="Foto tomada", use_column_width=True)

    # Preprocesar imagen (cambia el tamaño al usado en tu modelo)
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    st.markdown(f"### La fruta es: **{class_names[predicted_class]}** con un {confidence*100:.2f}% de confianza")


