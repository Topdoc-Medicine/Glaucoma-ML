from Tkinter import *
import tkMessageBox as messagebox
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas
import h5py
import glob

h5file =  "CNN2Weights.h5"

with h5py.File(h5file,'r') as fid:
     model = load_model(fid)

def get_filenames():
    global path
    path = r"test"
    return os.listdir(path)

def autoroi(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=5)

    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = img[y:y+h, x:x+w]

    return roi


def prediction():
    list_of_files = glob.glob('glaucomadataset/glaucoma/*') #testing different files
    latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(latest_file)
    img = autoroi(img)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])

    prediction = model.predict(img)
    # Class = prob.argmax(axis=1) # why was this here lmao
    print(prediction)


    return(prediction)


finalPrediction = prediction()
print "Final Prediction = ", finalPrediction
if (finalPrediction == 1):
    print("Congratulations! You are healthy!")
else:
    print("Unfortunately, you have been diagnosed with glaucoma.")
