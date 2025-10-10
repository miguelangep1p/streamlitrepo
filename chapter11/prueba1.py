import streamlit as st
import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time

st.title("🧠 Entrenamiento de Red Neuronal con MNIST (OpenCV MLP)")

st.write("📥 Cargando dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int32)

# ==============================
# 🔽 Reducir el tamaño del dataset
# ==============================
sample_size = 5000  # <<< Solo 5000 muestras para entrenamiento rápido
X = X[:sample_size]
y = y[:sample_size]
st.info(f"📊 Usando una muestra reducida de {sample_size} imágenes del dataset MNIST.")

# ==============================
# 🔹 Dividir dataset
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# 🔹 One-hot encoding
# ==============================
y_train_one_hot = np.zeros((y_train.size, 10), np.float32)
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# ==============================
# 🧠 Configuración de la red
# ==============================
st.write("🔧 Configurando red neuronal...")
ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([784, 128, 64, 10]))  # Más simple para muestra pequeña
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01, 0.9)
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 1000, 1e-4))

X_train = np.array(X_train, dtype=np.float32)
y_train_one_hot = np.array(y_train_one_hot, dtype=np.float32)

# Crear el objeto TrainData
train_data = cv2.ml.TrainData_create(
    samples=X_train,
    layout=cv2.ml.ROW_SAMPLE,
    responses=y_train_one_hot
)

# ==============================
# 🚀 Entrenamiento con progreso
# ==============================
st.write("🔄 Entrenando la red neuronal (esto tomará un momento)...")
progress_bar = st.progress(0)

# Entrenamiento en pasos simulados
num_steps = 10
for i in range(num_steps):
    ann.train(train_data, flags=cv2.ml.ANN_MLP_UPDATE_WEIGHTS if i > 0 else 0)
    progress_bar.progress(int((i + 1) / num_steps * 100))
    time.sleep(0.3)  # Simula avance visual

# ==============================
# 📊 Evaluación del modelo
# ==============================
st.write("📈 Evaluando precisión del modelo...")
X_test = np.array(X_test, dtype=np.float32)
_, y_pred = ann.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = np.mean(y_pred == y_test) * 100

st.success(f"✅ Precisión con MNIST (muestra 5000): {accuracy:.2f}%")

# ==============================
# 💾 Guardar modelo
# ==============================
ann.save("modelo_mnist_opencv_reducido.xml")
st.info("💾 Modelo guardado como 'modelo_mnist_opencv_reducido.xml'")
