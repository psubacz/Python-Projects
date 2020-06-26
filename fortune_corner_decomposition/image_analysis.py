import numpy as np
import cv2
from scipy.spatial import Voronoi, voronoi_plot_2d
import random as rd
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''
https://en.wikipedia.org/wiki/Fortune%27s_algorithm
https://jacquesheunis.com/post/fortunes-algorithm/
https://jacquesheunis.com/post/fortunes-algorithm/#fn:1
'''

#import keras
eq = deque()

#Load the image
IMAGE_PATH = 'mountian.jpeg'

from sklearn.cluster import KMeans
img = cv2.imread(IMAGE_PATH)
HEIGHT,WIDTH,DEPTH = img.shape

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.249855)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
h = []
w = []
for hs in range(0,HEIGHT):
    for ws in range(0,WIDTH):
        if dst[hs][ws] > 127:
            h.append(hs)
            w.append(ws)
            
X = np.array(list(zip(h, w))).reshape(len(h), 2)
K = 30
kmeans_model = KMeans(n_clusters=K).fit(X)
print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)
#x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
#x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
#
#kmeans_model.cluster_centers_

#for index in range(HEIGHT):
#    line = dst[index,0:WIDTH]
#    new_site = np.where(line > 127)
#    if not (len(*new_site) >0):
#        continue
#    for site in new_site[0]:
#        points.append((index,site))
#Kmean = KMeans(n_clusters=z).fit(dst)

#points=[]

#kernal_size= 23

#        K = dst[h:h+kernal_size,w:w+kernal_size]
#        
#        
#        k = cv2.waitKey(30)
#        if k == 27:
#            break
#        
       
        
        
## KMeans algorithm 
#kmeans_model = KMeans(n_clusters=len(points)).fit(dst)
#plt.plot()
#for i, l in enumerate(kmeans_model.labels_):
#    print(i,l)
##plt.show()

#

##blur = cv2.GaussianBlur(dst,(5,5),0)
##ret,thresh4 = cv2.threshold(blur,254,255,cv2.THRESH_TOZERO)

#eq.append('x')
#
#

#
#print(len(points))
#
#     while (len(new_site)>0):
#         print(new_site[0])
#         new_site.pop()
#
#
#
# # While the event queue still has items in it:
# while len(list(eq)) != 0:
#     if x_line >HEIGHT:
#         break
#
#    
#
#     cv2.line(dst,(0,x_line),(WIDTH,x_line+1),(255,255,255),3)1
#    
#     
#    
#    If the next event on the queue is a site event:
#    
#        Add the new site to the beachline
#    
#    Otherwise it must be an edge-intersection event:
#    
#        Remove the squeezed cell from the beachline
#    
#    Cleanup any remaining intermediate state
