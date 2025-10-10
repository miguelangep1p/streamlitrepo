import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread('images/persona.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar las esquinas usando goodFeaturesToTrack
corners = cv2.goodFeaturesToTrack(gray, maxCorners=7, qualityLevel=0.05, minDistance=25)
corners = np.float32(corners)

# Dibujar círculos en las características detectadas
for item in corners:
    x, y = item[0]
    x, y = int(x), int(y)  # Convertir las coordenadas a enteros
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Color azul (BGR)

# Mostrar la imagen con los puntos dibujados
cv2.imshow("Top 'k' features",cv2.resize(img, (400,520)))
cv2.waitKey(0)
cv2.destroyAllWindows()
