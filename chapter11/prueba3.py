import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2

# Cargar el conjunto de datos MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocesar los datos
X_train = X_train.astype('float32') / 255.0  # Normalizar a [0, 1]
X_test = X_test.astype('float32') / 255.0

# Añadir una dimensión para el canal de color (grayscale)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Definir el modelo como una CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Capa convolucional
    layers.MaxPooling2D(pool_size=(2, 2)),  # Capa de max pooling
    layers.Conv2D(64, (3, 3), activation='relu'),  # Capa convolucional
    layers.MaxPooling2D(pool_size=(2, 2)),  # Capa de max pooling
    layers.Flatten(),  # Aplanar la salida
    layers.Dense(128, activation='relu'),  # Capa oculta
    layers.Dense(10, activation='softmax')  # Capa de salida para 10 clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))  # Aumenta las épocas si es necesario

# Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Precisión en el conjunto de prueba:', test_acc)

# Guardar el modelo
model.save('modelo_mnist_cnn.h5')

# Probar con una nueva imagen
img = cv2.imread('./images/prueba0.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

# Hacer una predicción
pred = model.predict(img)
predicted_digit = np.argmax(pred)
print("Dígito predicho:", predicted_digit)
