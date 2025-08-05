import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

# Lista de frutas
frutas = ["manzana", "banana", "kiwi", "naranja", "pina", "uva", "fresa", "sandia"]

# Base del dataset
base_dataset_path = "dataset_bn"

# Verificar rutas con subcarpetas internas y mostrar conteo de imágenes
for fruta in frutas:
    ruta = os.path.join(base_dataset_path, fruta)
    if not os.path.exists(ruta):
        print(f"❌ Carpeta no encontrada: {ruta}")
    else:
        imagenes = os.listdir(ruta)
        print(f"✅ Carpeta encontrada: {ruta} - {len(imagenes)} imágenes")

# Crear generadores de datos
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=15,
    zoom_range=[0.5,1.5],
    validation_split=0.2
)

data_gen_entrenamiento = datagen.flow_from_directory(
    base_dataset_path,
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    subset="training"
)

data_gen_pruebas = datagen.flow_from_directory(
    base_dataset_path,
    target_size=(224,224),
    batch_size=32,
    shuffle=True,
    subset="validation"
)

# Crear modelo MobileNetV2
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(frutas), activation="softmax")(x)

modelo = tf.keras.Model(inputs, outputs)

modelo.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# Entrenar modelo
EPOCAS = 50
historial = modelo.fit(
    data_gen_entrenamiento,
    epochs=EPOCAS,
    validation_data=data_gen_pruebas
)

# Guardar modelo
os.makedirs("model", exist_ok=True)
modelo.save("model/fruits_model.h5")

# Graficar resultados
acc = historial.history["accuracy"]
val_acc = historial.history["val_accuracy"]
loss = historial.history["loss"]
val_loss = historial.history["val_loss"]
rango_epocas = range(EPOCAS)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(rango_epocas, acc, label="Entrenamiento")
plt.plot(rango_epocas, val_acc, label="Validación")
plt.title("Precisión")
plt.legend()

plt.subplot(1,2,2)
plt.plot(rango_epocas, loss, label="Entrenamiento")
plt.plot(rango_epocas, val_loss, label="Validación")
plt.title("Pérdida")
plt.legend()
plt.show()
