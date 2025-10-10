import cv2
import numpy as np

img = cv2.imread('images/persona.jpg')
num_rows, num_cols = img.shape[:2]

translation_matrix = np.float32([ [1,0,int(0.5*num_cols)], [0,1,int(0.5*num_rows)] ])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)

img_translation = cv2.warpAffine(img, translation_matrix, (2*num_cols, 2*num_rows))
img_rotation = cv2.warpAffine(img_translation, rotation_matrix, (num_cols*2, num_rows*2))

# Redimensionar la imagen a 400x520
img_rotation_resized = cv2.resize(img_rotation, (400, 520))

# Crear la ventana y redimensionarla
cv2.namedWindow('Rotacion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Rotacion', 400, 520)

# Mostrar la imagen redimensionada
cv2.imshow('Rotacion', img_rotation_resized)
cv2.waitKey()