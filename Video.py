# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:27:24 2019

@author: Gabriel
"""

import os
from os import listdir
import matplotlib.pyplot as plt
import time
import cv2
import imutils
from VideoFunctions import VideoFunctions
from ModelFunctions import ModelFunctions
from keras.models import load_model
from os import listdir
from keras import backend as K
from imutils import build_montages
from pyimagesearch.minivggnet import MiniVGGNet
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.utils import np_utils


###############################################################################
#####                                                                     ##### 
#####                    CHECK VIDEO LOGO                                 ##### 
#####                                                                     #####                                            
###############################################################################
#Load Model
model=ModelFunctions.loadModel("Everis2")
#Load labels
labelNames = ["Adidas", "Negatives"]

#Video
video='.\\videoadidas.mp4'
cap=cv2.VideoCapture(video)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(".//TestAdidas.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

#Numero Frame
NumFrame = 0
#Total Frames
TotalFrames=0
#Frames logo
FramesLogo =0
#Posiciones Frame con Logo
PositionFramesLogo=[]
#Array frame y boxes
GraphNumFrame = []
#Number of Boxes per Frame
GraphBoxesFrame=[]

while (cap.isOpened()):
    ret, frame = cap.read()
    (winW, winH) = (64, 64)
       
    i=0
    windows=[]
    boundingBoxes=[]
    # loop over the image pyramid
    for cont,resized in enumerate(VideoFunctions.pyramid(frame, scale=1.5)):
    	# loop over the sliding window for each layer of the pyramid
    	for (x, y, window) in VideoFunctions.sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
    		# if the window does not meet our desired window size, ignore it
    		if window.shape[0] != winH or window.shape[1] != winW:
    			continue
    
    		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    		# WINDOW
            
    		black = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    		image2 = np.expand_dims(black, axis=2)
    		image2 = np.expand_dims(image2, axis=0)
    		image2 = image2.astype("float32") / 255.0
    		windows.append(black)
                       
    		i+=1           
            
    		probs = model.predict(image2)
    		prediction = probs.argmax(axis=1)

    		if probs[0][0] >= 0.95:
    			prediction[0]=0
    		else:
    			prediction[0]=1
            
    		label = labelNames[prediction[0]]
    
            
    		if prediction[0]==0:
    		    if cont==0:
    		        boundingBoxes.append((x,y,x+winW, y+winH))
    		    if cont==1:
    		        boundingBoxes.append(((x),(y),int((x+winW)*1.5),
                                    int((y+winH)*1.5)))
    		    if cont==2:
    		        boundingBoxes.append(((x),(y),int((x+winW)*2.25),
                                    int((y+winH)*2.25)))
    		    if cont==3:
    		        boundingBoxes.append(((x),(y),int((x+winW)*3.375),
                                    int((y+winH)*3.375)))                
                
    		color=(0,0,255)
            
    		if prediction[0]==0:
    			color=(0,255,0)
    
    output=np.stack(windows, axis=0)
    if len(boundingBoxes) > 0:
        boundingBoxes=np.stack(boundingBoxes, axis=0)
     
    	# perform non-maximum suppression on the bounding boxes
        pick = VideoFunctions.non_max_suppression_fast(boundingBoxes, 0.3)
        print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
     
    	# loop over the picked bounding boxes and draw them
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
     
        print(NumFrame) 
    #GRAPHS
    #Array de numero de frame y boxes
    GraphNumFrame.append(NumFrame)
    GraphBoxesFrame.append(len(pick))
    #Contar numero de frame
    NumFrame = NumFrame + 1
    TotalFrames= TotalFrames + 1
    if len(pick) > 0:
        FramesLogo = FramesLogo + 1
        PositionFramesLogo.append(NumFrame)
    #Saving frames    
    # Write the frame into the file 'output.avi'
    out.write(frame)
    # display the images
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

###############################################################################
#####                                                                     ##### 
#####                            GRAPHS                                   ##### 
#####                                                                     #####                                            
###############################################################################

colors = (0,0,0)
plt.scatter(GraphNumFrame, GraphBoxesFrame, c=colors)
plt.title('Scatter Graph of Adidas Boxes')
plt.xlabel('Number of Frame')
plt.ylabel('Number of boxes')
plt.show()