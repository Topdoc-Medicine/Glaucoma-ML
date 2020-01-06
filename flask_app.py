from flask import *
from datetime import *
from pytz import *

def get_pst_time():
    date_format='%m/%d/%Y %H:%M:%S'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    return pstDateTime


import os

app = Flask(__name__)

from keras.models import load_model
import cv2
import numpy as np
import h5py

# def get_filenames():
#     global path
#     path = r"test"
#     return os.listdir(path)

@app.route('/upload')
def upload():
    return render_template("index2.html")

diagnoses = []

@app.route('/past')
def go():
    ret = ""
    for item in diagnoses:
        ret = ret + item + "\n\n"
    return ret

@app.route('/', methods = ['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        h5file =  "/home/topdoc/mysite/dataweightsvha.h5"
        with h5py.File(h5file,'r') as fid:
            model = load_model(fid)
            Class = prediction(model, f.filename)
            diagnoses.clear()
            if (Class == 0):
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, it is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice.")
                return "It is likely that you have glaucoma. Please contact a medical professional as soon as possible for advice."
            else:
                today = str(get_pst_time())
                diagnoses.append(today + ": According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional.")
                return "According to our algorithm, you do not have glaucoma! If you have further questions, please contact a medical professional."

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


def prediction(m, file):
    # list_of_files = glob.glob('data/test/*')
    # latest_file = max(list_of_files, key=os.path.getctime)
    img = cv2.imread(file)
    img = autoroi(img)
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, [1, 256, 256, 3])

    prob = m.predict(img)
    Class = prob.argmax(axis=-1)

    return(Class)
