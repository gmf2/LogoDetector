# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:16:39 2019

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
#####                         CREATE MODEL                                ##### 
#####                                                                     #####                                            
###############################################################################
NUM_EPOCHS = 25
INIT_LR = 1e-2
BS = 16

imageLength=64
imageWidth=64

trainX=ModelFunctions.imgStack("train")
testX=ModelFunctions.imgStack("test")
trainY=ModelFunctions.labelStack("train")
testY=ModelFunctions.labelStack("test")

    
# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainX = trainX.reshape((trainX.shape[0], 1, imageLength, imageWidth))
	testX = testX.reshape((testX.shape[0], 1, imageLength, imageWidth))
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainX = trainX.reshape((trainX.shape[0], imageLength, imageWidth, 1))
	testX = testX.reshape((testX.shape[0], imageLength, imageWidth, 1))
 
# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels
#trainY = np_utils.to_categorical(trainY, 4)
#testY = np_utils.to_categorical(testY, 4)

# initialize the label names
labelNames = ["Adidas", "Negatives"]

#barrel, binocular, bonsai, dolphin

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
model = MiniVGGNet.build(width=imageWidth, height=imageLength, depth=1, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	batch_size=BS, epochs=NUM_EPOCHS)

# make predictions on the test set
preds = model.predict(testX)

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

modelName="Model1"
ModelFunctions.saveModel(model,modelName)

