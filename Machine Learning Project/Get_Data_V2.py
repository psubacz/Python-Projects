# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:07:23 2018

@author: Eastwind
"""

import numpy as np
import cv2
from grabscreen import grab_screen
import time
from getkeys import key_check
import os

a = [1,0,0]
d = [0,1,0]
nk = [0,0,1]

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output

file_name = 'training_data_200x150_.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
    print('Finished loading previous data')
else:
    print('File does not exist, starting fresh!')
    training_data = []
    
def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = True
    while(True):
        if not paused:
            
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,600))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (200,150))
            
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
#            print(output)
            training_data.append([screen,output])
            
            if len(training_data) % 1000 == 0:
                print('recording...')
                print(len(training_data))
                np.save(file_name,training_data)
#           #DEBUG     
            cv2.imshow('test',screen)
            if cv2.waitKey(25)& 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        

main()
