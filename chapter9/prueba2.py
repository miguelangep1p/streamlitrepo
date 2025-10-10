import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()  # Crear el detector SIFT
    descriptors_list = []  # Lista para almacenar los descriptores

    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if descriptors is not None:
            descriptors_list.append(descriptors)

    all_descriptors = np.vstack(descriptors_list)
    return all_descriptors

def create_visual_dictionary(all_descriptors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans.cluster_centers_

def main(image_folder, num_clusters):
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.jpg')]

    all_descriptors = extract_sift_features(image_paths)

    visual_dictionary = create_visual_dictionary(all_descriptors, num_clusters)

    print("Centroides del diccionario visual:")
    print(visual_dictionary)

    np.save('visual_dictionary.npy', visual_dictionary)

if __name__ == "__main__":
    image_folder = 'D:/UNT CICLOS/VI CICLO/SIST. INTELIGENTES/S4/opencv_interface/chapter9/imagenes/zapatos_formales'  
    num_clusters = 50  
    
    main(image_folder, num_clusters)

#CREAR DICCIONARIO VISUAL