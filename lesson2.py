# example of pixel normalization
from numpy import asarray
from PIL import Image
from sklearn import preprocessing

# load image
image = Image.open('bondi_beach.jpg')
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
pixels /= 255.0
# confirm the normalization
print('Normalized Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# Fit your data on the scaler object
pixels_stand = preprocessing.scale(pixels.flatten())
pixels_stand = pixels_stand.reshape(pixels.shape)
print('Standardized Min: %.3f, Max: %.3f' % (pixels_stand.min(), pixels_stand.max()))