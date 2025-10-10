import cv2 
import numpy as np 
 
img = cv2.imread('images/persona.jpg') 
img_resized = cv2.resize(img, (400, 520))
cv2.imshow('Original', img_resized) 
 
# generating the kernels 
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) 
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]]) 
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1], 
                             [-1,2,2,2,-1], 
                             [-1,2,8,2,-1], 
                             [-1,2,2,2,-1], 
                             [-1,-1,-1,-1,-1]]) / 8.0 
 
# applying different kernels to the input image 
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1) 
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2) 
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3) 
 
output1_resized = cv2.resize(output_1,(400,520))
output2_resized = cv2.resize(output_2,(400,520))
output3_resized = cv2.resize(output_3,(400,520))

cv2.imshow('Sharpening', output1_resized) 
cv2.imshow('Excessive Sharpening', output2_resized) 
cv2.imshow('Edge Enhancement', output3_resized) 
cv2.waitKey(0) 