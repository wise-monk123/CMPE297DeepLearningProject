# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import time

import threading

import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask, SkinMask (True|False) OR BkgrndSubMask ('x' key)
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0

banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    4- Exit	
    '''



def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )



def skinMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, visualize, mod, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        #res = cv2.UMat.get(res)
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res



def binaryMask(frame, x0, y0, width, height, framecount, plot ):
    global guessGesture, visualize, mod, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        #ores = cv2.UMat.get(res)
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res


# This is the new mask mode. It simply tries to remove the background content by taking a image of the 
# background and subtracts it from the new frame contents of the ROI window.
# So in order to use it correctly, keep the contents of ROI window stable and without your hand in it 
# and then press 'x' key. If you can see the contents of ROI window all blank then it means you are
# good to go for gesture prediction
def bkgrndSubMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, takebkgrndSubMask, visualize, mod, bkgrnd, saveImg
        
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
    #Take background image
    if takebkgrndSubMask == True:
        bkgrnd = roi
        takebkgrndSubMask = False
        print("Refreshing background image for mask...")		

    
    #Take a diff between roi & bkgrnd image contents
    diff = cv2.absdiff(roi, bkgrnd)

    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
    mask = cv2.GaussianBlur(diff, (3,3), 5)
    mask = cv2.erode(diff, skinkernel, iterations = 1)
    mask = cv2.dilate(diff, skinkernel, iterations = 1)
    res = cv2.bitwise_and(roi, roi, mask = mask)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
        #t.join()
        #myNN.update(plot)
        
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    
    return res
