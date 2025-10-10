import cv2 
import numpy as np 
 
img = cv2.imread('images/persona.jpg') 
 
img_gaussian = cv2.GaussianBlur(img, (13,13), 0) 
img_bilateral = cv2.bilateralFilter(img, 13, 70, 50) 
 
cv2.imshow('Input', cv2.resize(img,(400,520))) 
cv2.imshow('Gaussian filter', cv2.resize(img_gaussian,(400,520))) 
cv2.imshow('Bilateral filter', cv2.resize(img_bilateral,(400,520))) 
cv2.waitKey() 