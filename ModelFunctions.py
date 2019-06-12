# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:44:28 2019

@author: Gabriel
"""


import os
from os import listdir
import matplotlib.pyplot as plt
import time
import cv2
import imutils
import numpy as np
from VideoFunctions import VideoFunctions
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
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 16

imageLength=64
imageWidth=64
    
#MODEL
class ModelFunctions:
    
    def saveModel(model,name):
        # Introduciendo el modelo y el nombre del mismo se guarda en la carpeta models
        model.save(".//models//"+name+".h5")
        
        print("Saved model to disk") 
        
    def loadModel(name):
        # Carga el modelo y lo devuelve
        print(".//models//"+name+".h5")
        loaded_model=load_model(".//models//"+name+".h5")
        
        print("Loaded model from disk")
        
        return loaded_model
    
    #LABELS
    def imgStack(tipo):
        #Tipo indica si es train o test
        array=[]
        for i,nombre in enumerate(listdir(".//DatasetTwo_old//"+tipo)):
    #        if i==0:
    ##            if nombre[0:3]=="adi" or nombre[0:3]=="Adi" or nombre[0:3]=="Neg":
    #            if nombre[0:3]=="BMW" or nombre[0:3]=="Neg":
    #                path="DatasetTwo2//"+tipo+"//"+nombre
    #                img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #                aux=cv2.resize(img, (imageLength, imageWidth)) 
    #                array=[aux]
    #        else:
            if nombre[0:3]=="adi" or nombre[0:3]=="Adi" or nombre[0:3]=="Neg":
    #        if nombre[0:3]=="BMW" or nombre[0:3]=="Neg":
                path=".//DatasetTwo_old//"+tipo+"//"+nombre
                img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                a=cv2.resize(img, (imageLength, imageWidth)) 
                array.append(a)
        
        output=np.stack(array, axis=0)
        
        return output
    
    def labelStack(tipo):
        label=[]
          
        for i,nombre in enumerate(listdir(".//DatasetTwo_old//"+tipo)):
            if nombre[0:3]=="adi" or nombre[0:3]=="Adi":
    #        if nombre[0:3]=="BMW":
                label.append([1,0])
            if nombre[0:3]=="Neg":
                
                label.append([0,1])
    #        print(tipo+"   "+str(i))
        output=np.stack(label, axis=0)
        
        return output

