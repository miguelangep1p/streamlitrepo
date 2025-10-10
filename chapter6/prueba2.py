import sys
import cv2
import numpy as np

# Draw vertical seam on top of the image 
def overlay_vertical_seam(img, seam): 
    img_seam_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)]) 
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0) 
    return img_seam_overlay

# Compute the energy matrix from the input image 
def compute_energy_matrix(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
    abs_sobel_x = cv2.convertScaleAbs(sobel_x) 
    abs_sobel_y = cv2.convertScaleAbs(sobel_y) 
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0) 

# Find vertical seam in the input image 
def find_vertical_seam(img, energy): 
    rows, cols = img.shape[:2] 
    seam = np.zeros(img.shape[0]) 
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0, :] = np.zeros(img.shape[1]) 
    edge_to = np.zeros(img.shape[:2]) 

    for row in range(rows - 1): 
        for col in range(cols): 
            if col != 0 and dist_to[row + 1, col - 1] > dist_to[row, col] + energy[row + 1, col - 1]: 
                dist_to[row + 1, col - 1] = dist_to[row, col] + energy[row + 1, col - 1]
                edge_to[row + 1, col - 1] = 1 

            if dist_to[row + 1, col] > dist_to[row, col] + energy[row + 1, col]: 
                dist_to[row + 1, col] = dist_to[row, col] + energy[row + 1, col] 
                edge_to[row + 1, col] = 0 

            if col != cols - 1 and dist_to[row + 1, col + 1] > dist_to[row, col] + energy[row + 1, col + 1]: 
                dist_to[row + 1, col + 1] = dist_to[row, col] + energy[row + 1, col + 1] 
                edge_to[row + 1, col + 1] = -1 

    seam[rows - 1] = np.argmin(dist_to[rows - 1, :]) 
    for i in (x for x in reversed(range(rows)) if x > 0): 
        seam[i - 1] = seam[i] + edge_to[i, int(seam[i])] 

    return seam 

# Remove the input vertical seam from the image 
def remove_vertical_seam(img, seam): 
    rows, cols = img.shape[:2] 
    for row in range(rows): 
        for col in range(int(seam[row]), cols - 1): 
            img[row, col] = img[row, col + 1] 
    img = img[:, 0:cols - 1] 
    return img 

if __name__ == '__main__': 
    # Cargar imagen por defecto
    img_input = cv2.imread('./images/pelota.jpg')  # Cambia 'imagen_default.jpg' por tu imagen por defecto

    # Usar un número predeterminado de costuras si no se proporciona
    num_seams = 20  # Número de costuras a eliminar

    img = np.copy(img_input) 
    img_overlay_seam = np.copy(img_input) 
    energy = compute_energy_matrix(img) 

    for i in range(num_seams): 
        seam = find_vertical_seam(img, energy) 
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        img = remove_vertical_seam(img, seam) 
        energy = compute_energy_matrix(img) 
        print('Number of seams removed = ', i + 1) 

    # Redimensionar imágenes para mostrar
    max_width = 640
    max_height = 480
    img_input_resized = cv2.resize(img_input, (max_width, max_height), interpolation=cv2.INTER_AREA)
    img_overlay_seam_resized = cv2.resize(img_overlay_seam, (max_width, max_height), interpolation=cv2.INTER_AREA)
    img_resized = cv2.resize(img, (max_width, max_height), interpolation=cv2.INTER_AREA)

    cv2.imshow('Input', img_input_resized) 
    cv2.imshow('Seams', img_overlay_seam_resized) 
    cv2.imshow('Output', img_resized) 
    cv2.waitKey()
