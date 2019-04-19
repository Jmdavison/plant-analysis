import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

np.seterr(divide='ignore', invalid='ignore')
img = cv2.imread("3.8_4.7.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

r = cv2.selectROI(img)
mask = np.zeros(img.shape[:2], np.uint8)
mask[r[0]:r[0]+r[2],r[1]:r[1]+r[3]] = 1

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(histr,color = col)
plt.show()

