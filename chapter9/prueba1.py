import cv2 
import numpy as np 

class DenseDetector(): 
    def __init__(self, step_size=20, feature_scale=20, img_bound=20): 
        # Create a dense feature detector 
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
 
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints 

class SIFTDetector():
    def __init__(self):
        self.detector = cv2.xfeatures2d.SIFT_create()

    def detect(self, img):
        # Convert to grayscale 
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # Detect keypoints using SIFT 
        return self.detector.detect(gray_image, None)  
 
if __name__=='__main__': 
    # Cargar la imagen directamente
    input_image = cv2.imread('./images/pelota.jpg')  # Cambia esto al nombre de tu imagen

    # Verificar si la imagen se cargó correctamente
    if input_image is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Redimensionar la imagen
    input_image = cv2.resize(input_image, (640, 480), interpolation=cv2.INTER_AREA)
    
    input_image_dense = np.copy(input_image)
    input_image_sift = np.copy(input_image)
 
    # Detección de puntos clave usando el detector denso
    keypoints = DenseDetector(20, 20, 5).detect(input_image)
    # Dibuja los puntos clave sobre la imagen de entrada 
    input_image_dense = cv2.drawKeypoints(input_image_dense, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    # Muestra la imagen de salida 
    cv2.imshow('Dense feature detector', input_image_dense) 
 
    # Detección de puntos clave usando SIFT
    keypoints = SIFTDetector().detect(input_image)
    # Dibuja los puntos clave SIFT sobre la imagen de entrada 
    input_image_sift = cv2.drawKeypoints(input_image_sift, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    # Muestra la imagen de salida 
    cv2.imshow('SIFT detector', input_image_sift) 

    # Espera hasta que el usuario presione una tecla 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
