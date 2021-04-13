import numpy as np #for matrix operations
import matplotlib.image as mpimg #for plotting the image pixels as images
import matplotlib.pyplot as plt
import math 
import scipy.misc #for saving image in JPG format
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def contrastSketching(image,a=0, b=255):
    m,n = image.shape
    c = min(image[0])
    d = max(image[0])

    for i in range(1,m):
        c = min(c,min(image[i]))
        d = max(d,max(image[i]))

    for i in range(m):
        for j in range(n):
            image[i][j] = (image[i][j]-c)*(b-a)/(d-c) + a

    return image

def gammaCorrection(image, gamma = 2.2):
    m,n = image.shape

    for i in range(m):
        for j in range(n):
            image[i][j] = 255*((image[i][j]/255)**(1/gamma))

    return image

def histogramEqualisation(image, L = 256):
    m,n = image.shape

    counts = defaultdict(int)
    for i in range(m):
        for j in range(n):
            image[i][j] = int(round(image[i][j]))
            counts[image[i][j]] += 1

    for i in range(1,L):
        counts[i] += counts[i-1]

    for i in range(m):
        for j in range(n):
            image[i][j] = (L-1)*(counts[image[i][j]])/(m*n)
    return image

def unsharpMask(image, radius=1, k=1, vrange= None):
    
    blurred = gaussian_filter(image, sigma=radius,mode='reflect')

    result = image + (image-blurred)*k
    if vrange is not None:
        return np.clip(result, vrange[0], vrange[1], out=result)
    return result


#################################################################   
#################################################################   

# Image Import
img = mpimg.imread("img1.jpg")
img = rgb2gray(img)

#################################################################   
#################################################################
plt.gray()
new_img = contrastSketching(img)
scipy.misc.imsave('img1_improved.jpg', new_img)  
imgplot = plt.imshow(new_img)
plt.show()