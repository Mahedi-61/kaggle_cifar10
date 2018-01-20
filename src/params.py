"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
Description: This File Contains All Paths and Parameters 
"""

# image parameter
IMG_SIZE = 32
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)

NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000
NUM_TEST_SAMPLES =  300000

CLS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]



# training path (local maching + cloud) for cloud remove (..)
TRAIN_DATA = r"/input/train/"
TRAIN_LABELS= r"/input/trainLabels.csv"
#TEST_DATA = r"/input/test/"
TEST_DATA = r"/input/test_data.dat"



#trained model architecture and weight (remove . for cloud)
MODEL_SAVE_PATH = "/output/"
MODEL_MOUNT_PATH = ["/model_1/", "/model_2/"]

MY_TRAIN_MODEL = ["my_train_resnet_18_model.json"]
MY_TRAIN_MODEL_WEIGHT = ["my_train_resnet_18_model_weight.h5"]


MODEL_NAME = MY_TRAIN_MODEL[0]
MODEL_WEIGHT = MY_TRAIN_MODEL_WEIGHT[0]




#submission file path
#RESULT_FILE_PATH = ""                 #for local machine
RESULT_FILE_PATH = "/output/"          #for cloud
RESULT_FILE_NAME = "cifar10_sub3.csv"




def get_model_path_list(nb_model):
    model_path_list = []
    for i in range(0, nb_model):
        model_path_list.append(MODEL_MOUNT_PATH[i] + MY_TRAIN_MODEL[i])

    return model_path_list



def get_model_weight_list(nb_model):
    model_path_list = []
    for i in range(0, nb_model):
        model_path_list.append(MODEL_MOUNT_PATH[i] + MY_TRAIN_MODEL_WEIGHT[i])

    return model_path_list











