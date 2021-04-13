
from matplotlib import pyplot as plt
import numpy as np #for matrix operations
import matplotlib.image as img #for plotting the image pixels as images
import math 
import scipy.misc #for saving image in JPG format





def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


image = img.imread("img1.jpg")
image  = np.array(image)
gray = rgb2gray(image)
gray = gray.astype(int)
image = gray



hist = np.zeros((256))
cdf = np.zeros((256))
m,n = image.shape
sm = 0

for j in range(m):
    for k in range(n):
            hist[image[j][k]] += 1
            sm += image[j][k]
            

for j in range(256):
        cdf[j] = cdf[j-1] + hist[j]/(m*n)
            

for j in range(256):
         cdf[j]= cdf[j]*256
             
             
             
mean_ = 0
var_ = 0
std_ = 0
mean_ = sm/(m*n)
     
var_sum = 0  
energy = 0
entropy = 0
fourth_moment = 0
third_moment = 0

for j in range(m):
        for k in range(n):
            pq = image[j][k]- mean_
            third_moment += (pq*pq*pq)/(m*n)
            fourth_moment += (pq*pq*pq*pq)/(m*n)
            var_sum += (image[j][k]- mean_)*(image[j][k] - mean_)
var_ = var_sum/(m*n) 
std_ = math.sqrt(var_)

for j in range(256): 
        p = hist[j]/(m*n)
        energy += p*p
        if p != 0:
           entropy += -1*(p*math.log(p,2))
           
           
           



hist = hist.astype(int)
x = np.arange(256)


y1 = []
y2 = []
for j in range(256):
       y1 += [hist[j]]
       y2 += [cdf[j]]
 
plt.bar(x, height = y1)
plt.title("Histogram")
plt.savefig('histogram.jpg')
plt.show()

plt.bar(x, height = y2)
plt.title("Normalised CDF")
plt.savefig('cdf.jpg')
plt.show()

print("Mean: "  + str(mean_ ))
print("Standard Deviatoin: "  + str(std_ ))
print("Energy: "  + str(energy ))
print("Entropy: "  + str(entropy ))
print("Third Moment: "  + str(third_moment ))
print("Fourth Moment: "  + str(fourth_moment ))
           
        
        
        
   
        

        


