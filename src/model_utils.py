"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import params

from keras.models import model_from_json
from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)




def save_model(model):
    print("saving model.....")
    
    json_string = model.to_json()
    open(params.MODEL_SAVE_PATH + params.MODEL_NAME, 'w').write(json_string)

     


def read_model(model_path, model_weights_path, custom_obj):
    print("reading... stored model architecture and weight")

    json_string = open(model_path).read()
    model = model_from_json(json_string, custom_objects = custom_obj)
    model.load_weights(model_weights_path)
    return model




class LossHistory(Callback):
    def on_train_begin(self, batch, logs = {}):
        self.losses = []
        self.val_losses = []
        

    def on_epoch_end(self, batch, logs = {}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))




def set_early_stopping():
    return EarlyStopping(monitor = "val_loss",
                               patience = 10,
                               mode = "auto",
                               verbose = 2)




def set_model_checkpoint():
    return ModelCheckpoint(params.MODEL_SAVE_PATH + params.MODEL_WEIGHT,
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 1)





def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = np.sqrt(0.1),
                             patience = 5,
                             min_lr = 5e-6)






def show_loss_function(loss, val_loss, nb_epochs):
    plt.xlabel("Epochs ------>")
    plt.ylabel("Loss -------->")
    plt.title("Loss function")
    plt.plot(loss, "blue", label = "Training Loss")
    plt.plot(val_loss, "green", label = "Validation Loss")
    plt.xticks(range(0, nb_epochs)[0::2])
    plt.legend()
    plt.show()













