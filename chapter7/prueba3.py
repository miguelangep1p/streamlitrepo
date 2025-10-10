import sys 
import cv2 
import numpy as np 

# Extract all the contours from the image 
def get_all_contours(img): 
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0) 
    # Find all the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Extract reference contour from the image 
def get_ref_contour(img): 
    contours = get_all_contours(img)

    # Extract the relevant contour based on area ratio
    for contour in contours: 
        area = cv2.contourArea(contour) 
        img_area = img.shape[0] * img.shape[1] 
        if 0.05 < area / float(img_area) < 0.8: 
            return contour 

if __name__=='__main__': 
    # Cargar las imágenes de referencia y entrada
    img1 = cv2.imread('./images/futbol.jpg')  # Cambia esto al nombre de tu imagen de referencia
    img2 = cv2.imread('./images/pelota_2.jpg')  # Cambia esto al nombre de tu imagen de entrada

    # Verificar si las imágenes se cargaron correctamente
    if img1 is None or img2 is None:
        print("Error: No se pudo cargar alguna de las imágenes.")
        sys.exit(1)

    # Redimensionar las imágenes
    img1 = cv2.resize(img1, (640, 480), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (640, 480), interpolation=cv2.INTER_AREA)

    # Extract the reference contour 
    ref_contour = get_ref_contour(img1)

    # Extract all the contours from the input image 
    input_contours = get_all_contours(img2) 

    closest_contour = None
    min_dist = None
    contour_img = img2.copy()
    cv2.drawContours(contour_img, input_contours, -1, color=(0, 0, 0), thickness=3) 
    cv2.imshow('Contours', contour_img)

    # Finding the closest contour 
    for i, contour in enumerate(input_contours): 
        # Matching the shapes and taking the closest one using 
        ret = cv2.matchShapes(ref_contour, contour, 3, 0.0)
        print("Contour %d matchs in %f" % (i, ret))
        if min_dist is None or ret < min_dist:
            min_dist = ret 
            closest_contour = contour

    cv2.drawContours(img2, [closest_contour], 0, color=(0, 0, 0), thickness=3) 
    cv2.imshow('Best Matching', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
