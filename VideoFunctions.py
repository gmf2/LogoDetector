# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:40:53 2019

@author: Gabriel
"""

import os
from os import listdir
import matplotlib.pyplot as plt
import time
import cv2
import imutils
import numpy as np
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
#####                             FUNCTIONS                               ##### 
#####                                                                     #####                                            
###############################################################################
#SLIDING

class VideoFunctions:
	
	def pyramid(image, scale=1.5, minSize=(30, 30)):
    	# yield the original image
		yield image
    
    	# keep looping over the pyramid
		while True:
    		# compute the new dimensions of the image and resize it
			w = int(image.shape[1] / scale)
			image = imutils.resize(image, width=w)
    
    		# if the resized image does not meet the supplied minimum
    		# size, then stop constructing the pyramid
			if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
   			    break 
    		# yield the next image in the pyramid
			yield image
    
	def sliding_window(image, stepSize, windowSize):
    	# slide a window across the image
		for y in range(0, image.shape[0], stepSize):
			for x in range(0, image.shape[1], stepSize):
    			# yield the current window
   			    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    
    
    #NON MAXIMA SUPRESSION
	def non_max_suppression_slow(boxes, overlapThresh):
    	# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []
     
    	# initialize the list of picked indexes
		pick = []
     
    	# grab the coordinates of the bounding boxes
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
     
    	# compute the area of the bounding boxes and sort the bounding
    	# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
    	# keep looping while some indexes still remain in the indexes
    	# list
		while len(idxs) > 0:
    		# grab the last index in the indexes list, add the index
    		# value to the list of picked indexes, then initialize
    		# the suppression list (i.e. indexes that will be deleted)
    		# using the last index
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
			suppress = [last]
    		# loop over all indexes in the indexes list
			for pos in range(0, last):
    			# grab the current index
   			    j = idxs[pos]
     
    			# find the largest (x, y) coordinates for the start of
    			# the bounding box and the smallest (x, y) coordinates
    			# for the end of the bounding box
   			    xx1 = max(x1[i], x1[j])
   			    yy1 = max(y1[i], y1[j])
   			    xx2 = min(x2[i], x2[j])
   			    yy2 = min(y2[i], y2[j])
     
    			# compute the width and height of the bounding box
   			    w = max(0, xx2 - xx1 + 1)
   			    h = max(0, yy2 - yy1 + 1)
     
    			# compute the ratio of overlap between the computed
    			# bounding box and the bounding box in the area list
   			    overlap = float(w * h) / area[j]
     
    			# if there is sufficient overlap, suppress the
    			# current bounding box
   			    if overlap > overlapThresh:
   			        suppress.append(pos)
     
    		# delete all indexes from the index list that are in the
    		# suppression list
			idxs = np.delete(idxs, suppress)
     
    	# return only the bounding boxes that were picked
		return boxes[pick]
    
	def non_max_suppression_fast(boxes, overlapThresh):
    	# if there are no boxes, return an empty list
		if len(boxes) == 0:
			print("No boxes")
			return []
     
    	# if the bounding boxes integers, convert them to floats --
    	# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")
     
    	# initialize the list of picked indexes	
		pick = []
     
    	# grab the coordinates of the bounding boxes
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
     
    	# compute the area of the bounding boxes and sort the bounding
    	# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
     
    	# keep looping while some indexes still remain in the indexes
    	# list
		while len(idxs) > 0:
    		# grab the last index in the indexes list and add the
    		# index value to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
     
    		# find the largest (x, y) coordinates for the start of
    		# the bounding box and the smallest (x, y) coordinates
    		# for the end of the bounding box
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
     
    		# compute the width and height of the bounding box
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
     
    		# compute the ratio of overlap
			overlap = (w * h) / area[idxs[:last]]
     
    		# delete all indexes from the index list that have
			idxs = np.delete(idxs, np.concatenate(([last],
			    np.where(overlap > overlapThresh)[0])))
     
    	# return only the bounding boxes that were picked using the
    	# integer data type
		return boxes[pick].astype("int")