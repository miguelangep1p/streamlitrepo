import cv2
import numpy as np

# Cargar la imagen que se superpondrá
overlay_img = cv2.imread('./images/estrella.png', -1)

def overlay_image(frame, overlay_img, x, y, w, h):
    overlay_img_resized = cv2.resize(overlay_img, (w, h))
    alpha_s = overlay_img_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y:y+h, x:x+w, c] = (alpha_s * overlay_img_resized[:, :, c] +
                                   alpha_l * frame[y:y+h, x:x+w, c])

# Detección y superposición
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejemplo: Superponer la imagen en coordenadas aleatorias
    overlay_image(frame, overlay_img, 100, 100, 200, 200)

    cv2.imshow('AR Example', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
