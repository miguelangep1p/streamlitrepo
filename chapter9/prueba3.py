import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

def extract_sift_features(image_path):
    sift = cv2.SIFT_create()
    # Cargar la imagen
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y calcular descriptores
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return descriptors

# Uso
image_folder = 'D:/UNT CICLOS/VI CICLO/SIST. INTELIGENTES/S4/opencv_interface/chapter9/imagenes'  # Cambia esto a la ruta de tu carpeta

# Definir las carpetas de las categorías
categories = ['zapatos_formales', 'zapatos_deportivos']
features = []
labels = []

for category in categories:
    category_folder = os.path.join(image_folder, category)
    for image_name in os.listdir(category_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(category_folder, image_name)
            # Extraer características SIFT
            descriptors = extract_sift_features(image_path)
            if descriptors is not None:
                features.append(descriptors)
                labels.append(category)

# Combinar todos los descriptores en una sola matriz para ajustar el KMeans
all_descriptors = np.vstack(features)

# Ajustar KMeans
num_clusters = 50  # Puedes ajustar esto según tus necesidades
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(all_descriptors)

# Crear histogramas de características para cada imagen
def create_feature_histograms(descriptors, kmeans):
    # Predecir el índice del cluster para cada descriptor
    cluster_indices = kmeans.predict(descriptors)
    # Crear un histograma de frecuencias
    histogram, _ = np.histogram(cluster_indices, bins=np.arange(kmeans.n_clusters + 1), density=True)
    return histogram

# Convertir los descriptores de cada imagen en un histograma
feature_vectors = []
for image_descriptors in features:
    histogram = create_feature_histograms(image_descriptors, kmeans)
    feature_vectors.append(histogram)

# Convertir a un array NumPy
feature_vectors = np.array(feature_vectors)

# Asignar etiquetas a cada imagen
labels = [category for category in categories for _ in range(10)]  # Cambia esto si el número de imágenes varía

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

# Entrenar el modelo SVM
svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Guardar el modelo
with open('trained_svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Guardar el KMeans (codebook)
with open('codebook.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
