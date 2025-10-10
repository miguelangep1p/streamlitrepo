import cv2
import numpy as np

img = cv2.imread('images/persona.jpg')
rows, cols = img.shape[:2]
img_resized = cv2.resize(img, (400, 520))

src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])

affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
img_output_resized = cv2.resize(img_output, (400, 520))

cv2.imshow('Input', img_resized)
cv2.imshow('Output', img_output_resized)
cv2.waitKey()