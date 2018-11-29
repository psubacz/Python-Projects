# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:24:44 2018

@author: Eastwind
"""
##Import library dependancies
import matplotlib.pyplot as plt
import numpy as np
from grabscreen import grab_screen
import time, os, cv2
from directkeys import PressKey,ReleaseKey, A, D, Enter
from getkeys import key_check
#from collections import deque, Counter
#from statistics import mode,mean
from keras.models import model_from_json
#form keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

##Set Parameters
weightspath = True
debug = False
displyoutput = True
height = 200
width = 150

retryImg = cv2.imread('retry_button.PNG',0)
w, h = retryImg.shape[::-1]
##Declare functions
def DoNothing(holdtime):
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(holdtime)
    
def MoveLeft(holdtime):
    ReleaseKey(D)
    PressKey(A)
    time.sleep(holdtime)
    
def MoveRight(holdtime):
    ReleaseKey(A)
    PressKey(D)
    time.sleep(holdtime)

def UpdateThres(turn_left_threshs,turn_right_thresh,fwd_thresh):
    new_turn_left_thresh = float(input('Enter new LEFT thresh, current is: '+str(turn_left_threshs)+'\n'))
    new_turn_right_thresh = float(input('Enter new Right thresh, current is: '+str(turn_right_thresh)+'\n'))
    new_fwd_thresh = float(input('Enter new Right thresh, current is: '+str(fwd_thresh)+'\n'))
    return new_turn_left_thresh,new_turn_right_thresh,new_fwd_thresh

def UpdateKeyDelay(holdtime):
    newtime = float(input('Enter new key delay time, current is:', holdtime))
    return newtime

def ActivateRetry():
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(Enter)
    time.sleep(0.05)
    ReleaseKey(Enter)
    print('Restarting Game, Please Wait!')
    for i in list(range(3))[::-1]:
        print(i+1)
        time.sleep(1)
        
def DeployModel():     
    #Load the trained weights, if the wights are found then load model/wieghts
    if os.path.isfile('model_6.json'):
        print('Loading previous model')
        json_file = open('model_6.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model_6.h5")
        print("Loaded model from disk")
        print('Model Loaded')    
        model.summary()
        
        #Set local parameters
        turn_left_thresh = 0.55
        turn_right_thresh = 0.55
        fwd_thresh = 0.8
        holdtime = 0.05
        threshold = 0.75
        
        if displyoutput:
            ConfidenceLevels =[0.1,0.1,0.1]
            fig, ax = plt.subplots()
            ind = np.arange(1, 4)
            plt.show(block=False)  
            LeftConfidence,NothingConfidence,RightConfidence = plt.bar(ind, ConfidenceLevels)
            ax.set_xticks(ind)
            ax.set_xticklabels(['Move Left', 'Do Nothing', ' Move Right'])
            ax.set_ylim([0, 1])
            ax.set_ylabel('Confidence')
            ax.set_title('System Prediction')

        #Start paused to get the ready
        paused = True
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
            
        while(True):
            if not paused:
                # 800x600 windowed mode
                screen = grab_screen(region=(0,40,800,600))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                screen = cv2.resize(screen, (height,width))
                prediction_choice = model.predict([screen.reshape(-1,width,height,1)])[0]

                #Detect the retry button using template matching               
                res = cv2.matchTemplate(screen,retryImg,cv2.TM_CCOEFF_NORMED)
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(screen, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
                    ActivateRetry()
                    
                if debug: 
                    loc = np.where( res >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(screen, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
                    nscreen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Debug Screen',nscreen)
                    
#                    cv2.imshow('test',screen)
                    if cv2.waitKey(25)& 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
                
                if prediction_choice[0]>turn_left_thresh:
                    MoveLeft(holdtime)
                    if displyoutput:
                        LeftConfidence.set_facecolor('g')
                        RightConfidence.set_facecolor('b')
                        NothingConfidence.set_facecolor('b')
                elif prediction_choice[1]>turn_right_thresh:
                    MoveRight(holdtime)
                    if displyoutput:
                        LeftConfidence.set_facecolor('b')
                        RightConfidence.set_facecolor('g')
                        NothingConfidence.set_facecolor('b')
                elif prediction_choice[2]>fwd_thresh:
                    DoNothing(holdtime)
                    if displyoutput:
                        LeftConfidence.set_facecolor('b')
                        RightConfidence.set_facecolor('b')
                        NothingConfidence.set_facecolor('g')


                #Display the output as animated bargrapths                    
                if displyoutput:
                    LeftConfidence.set_height(prediction_choice[0])
                    RightConfidence.set_height(prediction_choice[1])
                    NothingConfidence.set_height(prediction_choice[2])
                    fig.canvas.draw_idle()
                    try:
                        # make sure that the GUI framework has a chance to run its event loop
                        # and clear any GUI events.  This needs to be in a try/except block
                        # because the default implementation of this method is to raise
                        # NotImplementedError
                        fig.canvas.flush_events()
                        
                    except NotImplementedError:
                        pass
                    
                    #Reset the graph colors for next loop!
                    LeftConfidence.set_facecolor('b')
                    RightConfidence.set_facecolor('b')
                    NothingConfidence.set_facecolor('b')
                    
            #Read the scan codes for manual input
            keys = key_check()
            
            #Pause the game
            if 'T' in keys:
                if paused:
                    paused = False
                    print('Unpaused!')
                    
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    
                    #Release all keys from being pressed while paused.
                    DoNothing(holdtime)
                    time.sleep(1)
                    
            # Update thresoholds for key presses
            if 'Y' in keys:
                turn_left_thresh,turn_right_thresh,fwd_thresh =UpdateThres(
                        turn_left_thresh,turn_right_thresh,fwd_thresh)
            if 'U' in keys:
               holdtime = UpdateKeyDelay(holdtime)
                
    else:
        print('No model weights found, exiting...') 
                
DeployModel()