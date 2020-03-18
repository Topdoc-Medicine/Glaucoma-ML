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

h5file =  "f1.h5"

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
    list_of_files = glob.glob('data/train/not_glaucoma/*') #testing different files
    latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(latest_file)
    img = autoroi(img)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3])

    prob = model.predict(img)
    print(prob) # only showing 2 outputs (1,0) when online most algorithms have many with decimals
    Class = prob.argmax(axis=1) #this might be the error line,
    return(Class)


Class = prediction()
print(Class) #unsure if 1 or 0 is healthy or not, giving same output for two different images
#always returning 0, used all 4 folders for tests
if (Class == 1):
    print("Congratulations! You are healthy!")
else:
    print("Unfortunately, you have been diagnosed with glaucoma.")
