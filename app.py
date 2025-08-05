import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

# Inicializar app Flask
app = Flask(__name__)

# Cargar modelo desde carpeta "model"
model = tf.keras.models.load_model("model/fruits_model.h5")

# Clases (en el mismo orden que se entrenó el modelo)
CLASES = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

# Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    ruta_imagen = None

    if request.method == "POST":
        archivo = request.files["imagen"]

        if archivo:
            # Generar nombre único y extensión correcta
            nombre_archivo = str(uuid.uuid4()) + os.path.splitext(archivo.filename)[1]

            # ✅ Crear carpeta si no existe
            os.makedirs("static/uploads", exist_ok=True)

            # Guardar archivo en la ruta
            ruta = os.path.join("static/uploads", nombre_archivo)
            archivo.save(ruta)

            # Procesar imagen
            img = Image.open(ruta).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            pred = model.predict(img_array.reshape(1, 224, 224, 3))
            indice = np.argmax(pred[0])
            prediccion = CLASES[indice]
            ruta_imagen = ruta

    return render_template("index.html", prediccion=prediccion, imagen=ruta_imagen)

# Ejecutar
if __name__ == "__main__":
    app.run(debug=True)
