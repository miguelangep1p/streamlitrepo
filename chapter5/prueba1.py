import cv2 
import numpy as np 
 
input_image = cv2.imread('images/persona.jpg')
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
 
# Version under opencv 3.0.0 cv2.FastFeatureDetector()
fast = cv2.FastFeatureDetector_create() 
 
# Detect keypoints 
keypoints = fast.detect(gray_image, None)
print("Number of keypoints with non max suppression:", len(keypoints)) 

 
# Draw keypoints on top of the input image 
img_keypoints_with_nonmax=input_image.copy()
cv2.drawKeypoints(input_image, keypoints, img_keypoints_with_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.imshow('FAST keypoints - with non max suppression', cv2.resize(img_keypoints_with_nonmax, (400,520)) )

# Disable nonmaxSuppression 
fast.setNonmaxSuppression(False) 
# Detect keypoints again 
keypoints = fast.detect(gray_image, None)  
print("Total Keypoints without nonmaxSuppression:", len(keypoints))
 
# Draw keypoints on top of the input image 
img_keypoints_without_nonmax=input_image.copy()
cv2.drawKeypoints(input_image, keypoints, img_keypoints_without_nonmax, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
cv2.imshow('FAST keypoints - without non max suppression', cv2.resize(img_keypoints_without_nonmax, (400,520)) )
cv2.waitKey() 