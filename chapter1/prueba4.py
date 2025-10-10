import cv2


# Cargar la imagen original
img = cv2.imread('./images/persona.jpg')

# Escalado usando interpolación lineal
img_scaled_linear = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

# Redimensionar la imagen escalada a 400x520
img_scaled_linear_resized = cv2.resize(img_scaled_linear, (400, 520))

# Mostrar la imagen redimensionada
cv2.imshow('Scaling - Linear Interpolation', img_scaled_linear_resized)

# Escalado usando interpolación cúbica
img_scaled_cubic = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# Redimensionar la imagen escalada a 400x520
img_scaled_cubic_resized = cv2.resize(img_scaled_cubic, (400, 520))

# Mostrar la imagen redimensionada
cv2.imshow('Scaling - Cubic Interpolation', img_scaled_cubic_resized)


img_scaled = cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)

cv2.waitKey()