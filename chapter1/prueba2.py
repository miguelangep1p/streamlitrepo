import cv2
import numpy as np

# Cargar la imagen original
img = cv2.imread('./images/persona.jpg')

# Obtener el número de filas y columnas
num_rows, num_cols = img.shape[:2]

# Aplicar la primera traslación
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70, num_rows + 110))

# Aplicar la segunda traslación
translation_matrix = np.float32([[1, 0, -30], [0, 1, -50]])
img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + 70 + 30, num_rows + 110 + 50))

# Redimensionar la imagen a 400x520
img_translation_resized = cv2.resize(img_translation, (400, 520))

# Crear la ventana y redimensionarla
cv2.namedWindow('Traslacion', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Traslacion', 400, 520)

# Mostrar la imagen redimensionada
cv2.imshow('Traslacion', img_translation_resized)

# Esperar a que se pulse una tecla para cerrar la ventana
cv2.waitKey()
cv2.destroyAllWindows()
