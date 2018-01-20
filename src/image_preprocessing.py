"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
Description: this file is for image preprocessing
"""

import numpy as np
import pandas as pd
import pickle, os
import PIL

from PIL import Image
from sklearn.preprocessing import LabelBinarizer

import params



def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x




def process_train_images():
    img_files = [params.TRAIN_DATA + str(i) +
                 ".png" for i in range(1, (params.NUM_TRAIN_SAMPLES + 1))]

    #print(img_files[:10])

    train_data = np.ndarray((params.NUM_TRAIN_SAMPLES,
                 params.IMG_SIZE, params.IMG_SIZE, params.IMG_CHANNEL),
                 dtype = np.float32)

    #Pillow returns numpy array of (width, height,channel(RGB))
    for i, img_file in enumerate(img_files):
        img = Image.open(img_file)
        img = img.resize((params.IMG_SIZE, params.IMG_SIZE), PIL.Image.BICUBIC)
        np_img = np.array(img)
        
        train_data[i] = preprocess_input(np_img)
        
        if (i % 5000 == 0):
            print("Processing {0} images".format(i))
    print("Processing {0} images completed".format(i + 1))
    
    df = pd.read_csv(params.TRAIN_LABELS)
    #print(df.head())

    train_label = df["label"].values
    #print(len(train_label))

    encoder = LabelBinarizer()
    train_label = encoder.fit_transform(train_label)

    #when full train is not used
    train_label = train_label[:params.NUM_TRAIN_SAMPLES]

    print("Train Data shape: ", train_data.shape)
    print("Train Label shape: ", train_label.shape)
    
    return train_data, train_label


#............. Saving and restoring data .......................
def load_train_data():
    print("Start Preprocessing train data")

    train_data, train_labels = process_train_images() # np array
    train_data = normalization(train_data)            # python list

    return train_data, train_labels  # np array, python list




def save_data(data, path):
    file = open(path, "wb")

    #serializing a Python object
    pickle.dump(data, file)  
    file.close()




def restore_data(path):
    file = open(path, "rb")
        
    #de-serializing a Python object
    data = pickle.load(file)
    return data





