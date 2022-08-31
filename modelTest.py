
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import os

from tensorflow.keras.models import load_model



test_data = "C:\RoundAboutDataTest"
labels = ['No Traffic Circle', 'Approaching', 'Entering','Inside','Exiting']

def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("model.h5")

temp = 'C:\\RoundAboutDataTest\\Approaching\\frame5vid11.mp4'
prediction = model.predict([prepare(temp)])
print(prediction)  # will be a list in a list.
print(labels[int(prediction[0][0])])



# for label in labels:
#     path = os.path.join(test_data, label)
#     success = 0
#     total = 0
#     for img in os.listdir(path):
#         data_dir = os.path.join(path, img)
#         sourceCurrent = test_data + "\\" + label+"\\"+img
#         print(sourceCurrent)
#         total+=1
#         prediction = model.predict([prepare(sourceCurrent)])
#         pred= labels[int(prediction[0][0])]
#         if(pred==label):
#             success+=1
#     print("success for "+label+" :"+success)
#     print("total for "+label+" :"+total)