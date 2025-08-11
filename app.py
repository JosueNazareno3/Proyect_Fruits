import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "modelo.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

st.title("Clasificador de Frutas 🍎🍌🍇")

uploaded_file = st.file_uploader("Sube una imagen de la fruta", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen (ajustar al tamaño que se usó en elentrenamiento)
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    st.write(f"### Es una **{class_names[predicted_class]}** 🍇 con un {confidence*100:.2f}% de confianza")
