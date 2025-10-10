import cv2
import numpy as np
import pickle

def extract_sift_features(image_path):
    sift = cv2.SIFT_create()
    # Cargar la imagen
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y calcular descriptores
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return descriptors

def create_feature_histograms(descriptors, kmeans):
    # Predecir el índice del cluster para cada descriptor
    cluster_indices = kmeans.predict(descriptors)
    # Crear un histograma de frecuencias
    histogram, _ = np.histogram(cluster_indices, bins=np.arange(kmeans.n_clusters + 1), density=True)
    return histogram

def load_model():
    # Cargar el modelo SVM y el modelo KMeans
    with open('trained_svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    with open('codebook.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    return svm_model, kmeans

def predict_image(image_path, svm_model, kmeans):
    descriptors = extract_sift_features(image_path)
    
    if descriptors is None:
        print("No se pudieron extraer descriptores de la imagen.")
        return None

    histogram = create_feature_histograms(descriptors, kmeans)
    histogram = histogram.reshape(1, -1)  # Asegúrate de que la forma sea correcta

    prediction = svm_model.predict(histogram)
    return prediction

if __name__ == "__main__":
    # Cargar el modelo
    svm_model, kmeans = load_model()

    # Ruta de la nueva imagen a clasificar
    new_image_path = './images/clasificador.jpg'  # Cambia esto a la ruta de tu imagen

    # Predecir la categoría de la imagen
    result = predict_image(new_image_path, svm_model, kmeans)
    
    if result is not None:
        print(f"La imagen es de la categoría: {result[0]}")
