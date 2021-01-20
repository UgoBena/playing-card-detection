import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
cards_values = ["2c","2s","4d","5d","5h","6s","7h","8c","8d","9h","10c","10s"]
MIN_MATCH_COUNT = 10
queryImage = cv.imread('5.jpg',0)          # queryImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with orb
kpQuery, desQuery = orb.detectAndCompute(queryImage,None)

for value in cards_values:
  card_image = cv.imread('cards/{}.png'.format(value),0) # trainImage
  kp2, des2 = orb.detectAndCompute(card_image,None)
  bf = cv.BFMatcher(cv.NORM_HAMMING)
  # Match descriptors.
  matches = bf.knnMatch(desQuery,des2,k=5)
  print(len(matches))
  if len(matches) > MIN_MATCH_COUNT:
    print("found the following card : {}".format(value))  

img=cv.drawKeypoints(queryImage,kpQuery,queryImage)
cv.imwrite("kp.png",img)
