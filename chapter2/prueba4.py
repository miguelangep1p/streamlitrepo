import cv2 
import numpy as np 
 
img = cv2.imread('images/persona.jpg', 0) 
img_resized = cv2.resize( img, (400,520) )
 
kernel = np.ones((5,5), np.uint8) 
 
img_erosion = cv2.erode(img, kernel, iterations=1) 
img_erosion_r= cv2.resize( img_erosion, (400,520) )
img_dilation = cv2.dilate(img, kernel, iterations=1) 
img_dilation_r = cv2.resize( img_dilation, (400,520) )

cv2.imshow('Input', img_resized) 
cv2.imshow('Erosion', img_erosion_r) 
cv2.imshow('Dilation', img_dilation_r) 
 
cv2.waitKey(0)