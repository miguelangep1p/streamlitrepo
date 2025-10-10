import sys 
import numpy as np
import cv2

if __name__ == '__main__':
    # Cargar la imagen predeterminada
    img = cv2.imread('./images/cuenca.jpg')  # Cambia esto al nombre de tu imagen
    if img is None:
        print("Error: La imagen no se pudo cargar.")
        sys.exit(1)

    # Redimensionar la imagen
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Eliminación de ruido
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    
    # Área de fondo seguro
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Encontrar el área de primer plano seguro
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Encontrar la región desconocida
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)

    # Añadir uno a todas las etiquetas para que el fondo seguro no sea 0, sino 1
    markers = markers + 1

    # Ahora, marcar la región desconocida con cero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 255, 255]

    # Mostrar resultados
    cv2.imshow('background', sure_bg)
    cv2.imshow('foreground', sure_fg)
    cv2.imshow('threshold', thresh)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
