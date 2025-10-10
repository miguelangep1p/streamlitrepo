import cv2 
import numpy as np 
 
img_emboss_input = cv2.imread('images/persona.jpg') 
img_emboss_input_resized = cv2.resize(img_emboss_input,(400,520))
# generating the kernels 
kernel_emboss_1 = np.array([[0,-1,-1], 
                            [1,0,-1], 
                            [1,1,0]]) 
kernel_emboss_2 = np.array([[-1,-1,0], 
                            [-1,0,1], 
                            [0,1,1]]) 
kernel_emboss_3 = np.array([[1,0,0], 
                            [0,0,0], 
                            [0,0,-1]]) 
 
# converting the image to grayscale 
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY) 
 
# applying the kernels to the grayscale image and adding the offset to produce the shadow
output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128 
output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128 
output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128 

output_1_resized = cv2.resize(output_1,(400,520))
output_2_resized = cv2.resize(output_2,(400,520))
output_3_resized = cv2.resize(output_3,(400,520))
 
cv2.imshow('Input', img_emboss_input_resized) 
cv2.imshow('Embossing - South West', output_1_resized) 
cv2.imshow('Embossing - South East', output_2_resized) 
cv2.imshow('Embossing - North West', output_3_resized) 
cv2.waitKey(0)