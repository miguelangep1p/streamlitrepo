import cv2 
import numpy as np 
 
img = cv2.imread('images/persona.jpg') 
img_resized = cv2.resize(img, (400, 520))
cv2.imshow('Original', img_resized) 
 
size = 15 
 
# generating the kernel 
kernel_motion_blur = np.zeros((size, size)) 
kernel_motion_blur[int((size-1)/2), :] = np.ones(size) 
kernel_motion_blur = kernel_motion_blur / size 
 
# applying the kernel to the input image 
output = cv2.filter2D(img, -1, kernel_motion_blur) 
output_resized = cv2.resize(output,(400,520))
 
cv2.imshow('Motion Blur', output_resized) 
cv2.waitKey(0) 