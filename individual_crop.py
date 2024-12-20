import cv2
import numpy as np

image = cv2.imread("data/ndvi.jpg")
image = cv2.resize(image, (image.shape[1]//3,image.shape[0]//3))

edged = cv2.Canny(image, 10, 200, True, 3)

cv2.imshow("edged", edged)
cv2.waitKey(0)

kern = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kern)
gradient_orig = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kern)
cv2.imshow("closed", gradient_orig)
cv2.waitKey(0)

(contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ix = 0
for c in contours:
    perim = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.02*perim, True)
    approx = cv2.convexHull(c)
    x,y,w,h = cv2.boundingRect(c)
    if( w>25 and h>25):
        ix += 1
        newImg = image[y:y+h, x:x+w]
        cv2.imwrite(str(ix) + ".png", newImg)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.drawContours(image, [rect], -1, (255,0,0), 2)

cv2.imshow("stuff", image)
cv2.waitKey(0)