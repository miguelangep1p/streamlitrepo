import cv2
import numpy as np
import streamlit as st

# Cargar el modelo
ann = cv2.ml.ANN_MLP_load('modelo_mnist_opencv_reducido.xml')

st.title("🔢 Reconocimiento de Dígitos con OpenCV")

# Subir imagen desde el usuario
archivo = st.file_uploader("Sube una imagen de un dígito manuscrito", type=["png", "jpg", "jpeg"])

if archivo is not None:
    # Leer la imagen
    file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
    imagen_prueba = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(imagen_prueba, caption="🖼 Imagen cargada", use_column_width=True)

    # Preprocesar
    imagen_prueba_resized = cv2.resize(imagen_prueba, (20, 20)).reshape(1, -1).astype(np.float32)
    imagen_prueba_resized = imagen_prueba_resized / 255.0

    # Predicción
    _, prediccion = ann.predict(imagen_prueba_resized)
    digito_predicho = np.argmax(prediccion)

    st.success(f"✅ Dígito predicho por el modelo: **{digito_predicho}**")
