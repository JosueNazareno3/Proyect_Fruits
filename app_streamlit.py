import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# ==============================
# CONFIGURACIÓN DEL TÍTULO
# ==============================
st.set_page_config(page_title="Clasificador de Frutas", page_icon="🍌", layout="centered")
st.title("🍎 Clasificador de Frutas con IA")
st.write("Sube una imagen de fruta y la IA intentará adivinar cuál es.")

# ==============================
# DESCARGAR MODELO SI NO EXISTE
# ==============================
MODEL_PATH = "model/fruits_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=TU_ID_DE_DRIVE"  # ← pon el ID del modelo aquí

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Descargando modelo..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ==============================
# CARGAR MODELO
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Lista de clases (ajusta a las tuyas)
class_names = ["banana", "manzana", "naranja", "uva", "sandía", "pera", "fresa", "piña"]

# ==============================
# SUBIR IMAGEN
# ==============================
uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesar imagen
    img_resized = image.resize((224, 224))  # ajusta según tu modelo
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    with st.spinner("Analizando imagen..."):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    # Mostrar resultado
    st.success(f"**Predicción:** {predicted_class} ({confidence:.2f}%)")
