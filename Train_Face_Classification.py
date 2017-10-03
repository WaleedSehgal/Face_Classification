from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2
import numpy as np
import pandas as pd
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Activation
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


#Function for image rotation
datagen = ImageDataGenerator(
        rotation_range=40,
        horizontal_flip=True,
        fill_mode='reflect')

#standardize image by resizing
def standardize_image(image, size=(100, 100)):
    return cv2.resize(image, size)
 

tr_df = pd.read_csv('E:/data/profile.csv')

#Image names
tr_U = tr_df['userid']
tr_U_ids = []
labels = []
dt = []


face_cascade = cv2.CascadeClassifier('E:/Anaconda3/pkgs/opencv3-3.1.0-py35_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    
for i in range(0,len(tr_U)):  
    img = load_img("D:/data1/training/"+str(tr_U[i])+".jpg")
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
        #shape (1, 3, 100, 100)
        img = img.reshape((1,) + img.shape)  
        ids = str(tr_U[i])
        tr_U_ids.append(ids)
        gen = tr_df.loc[tr_df['userid'] == ids, 'gender'].iloc[0]
        labels+= [gen]*6 
        
        i = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir='D:/pres/', save_format='jpeg'):
            dt.append(img)
            i += 1
            if i > 5:
                break  
               


data = np.array(dt) / 255.0

labels = np_utils.to_categorical(labels, 2)  

(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.30, random_state=2)

trainData = trainData[: ,0, :,:]
testData = testData[: ,0, :,:]


model = Sequential()

#The input here is 7500 pixel intensities--> 50*50*3
model.add(Convolution2D(32,3,3,input_shape=( 100, 100, 3), activation="relu" ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(384, init="uniform", activation="relu"))

model.add(Dense(2))
model.add(Activation("softmax"))

s_gd = SGD(lr=0.00001)

#since the labels are binary(0/1), we have used binary crossentropy loss function
model.compile(loss="binary_crossentropy", optimizer=s_gd,metrics=["accuracy"])


model.fit(np.array(trainData), np.array(trainLabels), nb_epoch=50, batch_size=1)

#Saves the model
model.save("D:/keras model/model.h5")
   
(loss, accuracy) = model.evaluate(testData, testLabels,batch_size=128, verbose=1)

print("[INFO] loss={:.2f}, accuracy: {:.2f}%".format(loss,accuracy * 100))

