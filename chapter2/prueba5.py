import cv2 
import numpy as np 
 
# Cargar la imagen
img = cv2.imread('images/persona.jpg') 

# Redimensionar la imagen a 400x520
img_r = cv2.resize(img, (400, 520))
rows, cols = img_r.shape[:2]  # Obtener las dimensiones de la imagen redimensionada
 
# Generar la máscara de viñeta usando kernels Gaussianos
kernel_x = cv2.getGaussianKernel(cols, 200) 
kernel_y = cv2.getGaussianKernel(rows, 200) 
kernel = kernel_y * kernel_x.T 
mask = 255 * kernel / np.linalg.norm(kernel) 

# Crear una copia de la imagen redimensionada para aplicar la máscara
output = np.copy(img_r) 

# Redimensionar la máscara al tamaño de la imagen
mask_resized = cv2.resize(mask, (cols, rows))

# Aplicar la máscara a cada canal de la imagen redimensionada
for i in range(3): 
    output[:,:,i] = output[:,:,i] * mask_resized  # Usar la máscara redimensionada
 
# Mostrar la imagen original y la imagen con el efecto de viñeta
cv2.imshow('Original', img_r) 
cv2.imshow('Vignette', output) 

cv2.waitKey(0)
cv2.destroyAllWindows()
