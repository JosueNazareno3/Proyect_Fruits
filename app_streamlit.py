# app_streamlit.py
import os
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Clasificador de frutas", layout="centered")

# --- Config: ruta al modelo ---
MODEL_LOCAL_PATH = "model/fruits_model.h5"  # ajusta si lo descargas
# Si prefieres descargar el modelo automáticamente, ver más abajo

@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"Modelo no encontrado en: {ruta_modelo}. Súbelo al repo o descarga al iniciar.")
    model = tf.keras.models.load_model(ruta_modelo)
    return model

CLASES = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

st.title("📷 Clasificador de frutas - Streamlit")
st.write("Sube una imagen y el modelo te dirá la fruta.")

# Cargar modelo (muestra error si no existe)
try:
    model = cargar_modelo(MODEL_LOCAL_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

uploaded = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    # Guardar una copia temporal (opcional)
    nombre = str(uuid.uuid4()) + os.path.splitext(uploaded.name)[1]
    ruta_tmp = os.path.join("temp_uploads")
    os.makedirs(ruta_tmp, exist_ok=True)
    ruta = os.path.join(ruta_tmp, nombre)
    with open(ruta, "wb") as f:
        f.write(uploaded.getbuffer())

    # Mostrar imagen
    st.image(ruta, caption="Imagen subida", use_column_width=True)

    # Preparar para predecir
    try:
        img = Image.open(ruta).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        pred = model.predict(img_array.reshape(1, 224, 224, 3))
        indice = int(np.argmax(pred[0]))
        prob = float(np.max(pred[0]))
        st.success(f"Predicción: **{CLASES[indice]}**  — confianza: {prob:.2f}")
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
