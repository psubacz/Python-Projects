# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:26:01 2018

@author: Eastwind
"""

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

training_data = np.load('data\\Final_Left.npy')

for data in training_data:
        img = data[0]
        choice = data[1]
        cv2.imshow('test',img)
        print(choice)
        if cv2.waitKey(25)& 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break