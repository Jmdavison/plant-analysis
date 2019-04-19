import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/ndvi.jpg')

print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.imshow(gray, cmap='gray')
# plt.show()

print(gray[0:100])

mask = np.zeros(img.shape[:2],np.uint8)

bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

rect = (0,0, img.shape[1] - 1, img.shape[0] - 1)

cv2.grabCut(img,mask,rect,bgModel,fgModel,1,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()

