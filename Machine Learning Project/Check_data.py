# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:26:01 2018

@author: Eastwind
"""

import numpy as np
from random import shuffle
import cv2

training_data = np.load('data\\Final_Nothing.npy')

shuffle(training_data)
lefts = []
rights = []
nothing = []

for data in training_data:
    img = data[0]
    choice = data[1]
    
    if choice == [1,0,0]:
        lefts.append([img,choice])
        
    elif choice == [0,1,0]:
        rights.append([img,choice])
        
    elif choice == [0,0,1]:
        nothing.append([img,choice])
        
lefts = lefts[:len(lefts)]
rights = rights[:len(rights)]
nothing = nothing[:len(nothing)]
final_data = lefts+rights+nothing

shuffle(final_data)

print('lefts',len(lefts),'rights',len(rights),'nothing',len(nothing),'final data',len(final_data))

#np.save('training_data_V2.npy',final_data)

#Data debuging, view the actual data...
for data in training_data:
#    time.sleep(0.2)
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(img.shape)
    print(choice)
    if cv2.waitKey(25)& 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

