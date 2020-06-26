#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:56:07 2020

@author: eastwind
"""
import numpy as np
import hashlib

class cluster:
    '''
    Class to descibe hold cluster center and calculuate geometric mean
    '''
    def __init__(self,index,y,x):
        self.index = index
        self.center = [y,x]
        self.old_center = None
        self.y_points = []
        self.x_points = []
        
    def get_info(self):
        return self.index,self.center
    
    def calculate_average_center(self):
        #Average the assigned points
        self.old_center = self.center
        self.center[0] = int(np.average(self.y_points))
        self.center[1] = int(np.average(self.x_points))
        self.y_points = []
        self.x_points = []
        
class pixel_kmeans:
    '''
    Kmeans clustering to find the cenetroids of a grouping of pixels
    '''
    def __init__(self,data_shape,number_of_clusters=1,seed_clusters = None,
                 threshold = 0.001,max_iters = 100,max_distance = 800,
                 eplison = None):
        '''
        args:
            number_of_clusters - integer of cluster to start with
            data_shape- tuple of integers of a pixel shape (HEIGHT,WIDTH,etc...)
            seed_clusters - Clusters that are precomputed
            max_iters: maximum iterations the al
            max_distance: the maximum euclidean distance away that a cluster can be assigned a point (float)
            eplison: Distance decay a cluster can include a new cluster (float)
        '''
        self.max_iters = max_iters
        self.threshold = threshold
        self.number_of_clusters = number_of_clusters
        self.clusters = []
        self.eplison = max_distance/(max_iters*2)
        self.max_distance = max_distance
        rand_y = np.random.randint(data_shape[0], size=(number_of_clusters))  # rand
        rand_x = np.random.randint(data_shape[1], size=(number_of_clusters))  #
        
        #Generate number of clusters
        if seed_clusters == None:
            for i in range(number_of_clusters):
                self.clusters.append(cluster(len(self.clusters),
                                         rand_y[i],
                                         rand_x[i]))
        
#        for cluster_ in self.clusters:
#            cluster_.get_info()
                
    def euclidean_distance(self,A,B):
        '''
        Vectorized calculation of L2 norm. Returns the euclidean distance
        '''
        return np.sqrt(np.sum((A - B)**2))
    
    def find_closest_cluster(self,data,data_size):
        '''
        
        Calculates the cluster distances
        
        '''
        clster_cnt = len(self.clusters) #number of clusters
        assgn_clust =  np.zeros((data_size[1],1))
        #get the distances to each cluster, encoded (clusters,y_pixels)
        c_dist = np.zeros((len(self.clusters),data_size[1]))
#        new_clster_ = np.zeros((clusters.shape[0],data_size[0]))
        for h in range(data_size[0]):
            for w in range(data_size[1]):
                center_dist_old = data_size[0]*data_size[1]
                for cluster_ in self.clusters:
                    assign_cluster= None
                    if data[h][w]>128:
                        center_dist = self.euclidean_distance(cluster_.center,np.array([h,w]))
                        if center_dist <center_dist_old:
                            assign_cluster,_ = cluster_.get_info()
                            
                if assign_cluster is not None:
                    #add point to supporting points
                    if center_dist > self.max_distance:
                         # Assign to new cluser
                        self.clusters.append(cluster(len(self.clusters), h, w))
                    else:
                        # Assign to known cluser
                        self.clusters[assign_cluster].y_points.append(h)
                        self.clusters[assign_cluster].x_points.append(w)
#                        self.clusters[assign_cluster].points.append(center_dist)
                else:
                    #do nothing
                    pass
        
        #Calcualte the new center of each cluster
        for cluster_ in self.clusters:
            if len(cluster_.y_points)>0:
                cluster_.calculate_average_center()
        print(self.number_of_clusters)
        
        
    def fit(self,data):
        '''
        fit the model 
    
        args:
            data:
            distance_metric: generator to calculate distance
        '''
            
        #Define initial threshold
        d_theta = self.threshold*2
        #Iteration counter
        iters = 0
        #run until cluter date is less than threshold meaning that the clusters no longer move
        while(d_theta > self.threshold):
            #Increment the execution counter
            iters +=1
            #If the max number of runs has exceeded the set value, return failure
            if (iters>=self.max_iters):
                print('Exceeded number of runs...')
#                return theta_j,assgn_clust
            else:
                #Calculate the L2 distance to each cluster. Returns assign data to closest clusters and new 
                #cluster centroids
                self.find_closest_cluster(data,data.shape)  
                
                #check to see if data is still moving. 
                for cluster_ in self.clusters:
                    if cluster_.old_center is not None:
                        d_theta = self.euclidean_distance(cluster_.old_center,cluster_.center)
                        print(d_theta)
                    #deal with empty clusters (empty)
                    
        print('Jobs done!')
                
import cv2
IMAGE_PATH = 'mountian.jpeg'
img = cv2.imread(IMAGE_PATH)
HEIGHT,WIDTH,DEPTH = img.shape
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.249855)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)
pk = pixel_kmeans(data_shape = (1920,1080),number_of_clusters=3)
pk.fit(dst)