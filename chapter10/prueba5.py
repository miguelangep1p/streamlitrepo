import cv2
import numpy as np

# Definir los vértices de la pirámide en 3D
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], dtype=np.float32)

# Definir las caras de la pirámide
faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2, 3]], dtype=object)

# Capturar la imagen de la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Parámetros de la cámara
    camera_matrix = np.array([[1000, 0, frame.shape[1] / 2], 
                               [0, 1000, frame.shape[0] / 2], 
                               [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))  # Sin distorsión
    rvec = np.array([0, 0, 0], dtype=np.float32)  # Rotación
    tvec = np.array([0, 0, 3], dtype=np.float32)  # Traslación

    # Proyectar los puntos 3D a 2D
    image_points, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    image_points = np.int32(image_points).reshape(-1, 2)
    
    # Dibujar las caras de la pirámide con colores
    for i, face in enumerate(faces[:-1]):  # Excluir la base
        color = (0, 255, 0) if i == 0 else (0, 0, 255) if i == 1 else (255, 0, 0) if i == 2 else (0, 255, 255)
        cv2.polylines(frame, [image_points[face]], isClosed=True, color=color, thickness=2)
    
    # Dibujar la base de la pirámide con un color diferente
    cv2.polylines(frame, [image_points[faces[-1]]], isClosed=True, color=(255, 255, 0), thickness=2)

    cv2.imshow('Piramide en 3D con Color y Movimiento', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Actualizar la posición de la pirámide
    tvec[2] += 0.01  # Mover la pirámide hacia adelante
    if tvec[2] > 5:
        tvec[2] = 3  # Resetear la posición

cap.release()
cv2.destroyAllWindows()