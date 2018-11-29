# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:31:55 2018

@author: Eastwind
"""

# import numpy as np
# from PIL import ImageGrab
# import cv2
# import time

# def screen_record(): 
    # last_time = time.time()
    # while(True):
        # # 800x600 windowed mode
        # printscreen =  np.array(ImageGrab.grab(bbox=(0,40,480,270)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        # last_time = time.time()
        # cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        # if cv2.waitKey(25) & 0xFF == ord('q'):
            # cv2.destroyAllWindows()
            # break
# screen_record()

# Done by Frannecklp

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


