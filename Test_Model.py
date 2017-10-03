# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:21:36 2017

@author: Waleed
"""

from keras.preprocessing.image import  load_img, img_to_array
import cv2
import numpy as np
import pandas as pd
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from keras.utils import np_utils
import sys


file = sys.argv[1]

def standardize_image(image, size=(100, 100)):
    return cv2.resize(image, size)
 

tr_df = pd.read_csv(file+'profile/profile.csv')

#Image names
tr_U = tr_df['userid']
tr_U_ids = []
testLabels = []
dt = []
n_d = pd.DataFrame()

face_cascade = cv2.CascadeClassifier('C:/Users/Waleed/Anaconda3/pkgs/opencv3-3.1.0-py35_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    
for i in range(0,len(tr_U)):  
#for i in range(0,10):
    img = load_img("D:/data1/test/"+str(tr_U[i])+".jpg")
    img = np.array(img, dtype='uint8')
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if (len(faces))>1: continue
 

    for f in faces:
        x, y, w, h = [ v for v in f ]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        sub_face = img[y:y+h, x:x+w]
        img = img_to_array(sub_face)
        img = standardize_image(img)
        dt.append(img)
        ids = str(tr_U[i])
        tr_U_ids.append(ids)
        gen = tr_df.loc[tr_df['userid'] == ids, 'gender'].iloc[0]
        testLabels.append(gen)

        

testData = np.array(dt) / 255.0

testLabels = np.array(testLabels)


#print(trainData)
model = load_model('D:/Keras model/model.h5')

predict = model.predict_classes(testData)


print(tr_U_ids)
print(predict)
print(len(tr_U_ids))
print(len(predict))
n_d = pd.DataFrame({'userid':tr_U_ids, 'gender':predict})



