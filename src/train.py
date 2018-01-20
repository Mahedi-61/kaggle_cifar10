"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
Description: Train a Resnet-18 model for cifar10 dataset
"""

# loading packages
import numpy as np
import os
import image_preprocessing
import my_model
import model_utils


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# training parameters
batch_size = 32
epochs = 50
nb_classes = 10

# Loading data
X_train, y_train = image_preprocessing.process_train_images()
(X_val, y_val) = (X_train[40000:50000], y_train[40000:50000])
X_train = X_train[:40000]
y_train = y_train[:40000]



# constructing model architecture 
model = my_model.my_resent_18(input_shape = X_train.shape[1:],
                              nb_classes = nb_classes)

model.compile(optimizer = Adam(lr = 1e-3),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])


# training 
history        = model_utils.LossHistory()
early_stopping = model_utils.set_early_stopping()
model_cp       = model_utils.set_model_checkpoint()
reduce_lr      = model_utils.set_reduce_lr()
callbacks = [history, early_stopping, model_cp, reduce_lr]


print("using real time data augmentation")
datagen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.1,
    horizontal_flip = True)

datagen.fit(X_train)


model.fit_generator(
    datagen.flow(X_train, y_train, batch_size = batch_size),
    steps_per_epoch = int(np.ceil(X_train.shape[0] / float(batch_size))),
    validation_data = (X_val, y_val),
    verbose = 2,
    epochs = epochs,
    callbacks = callbacks)


#saving model
model_utils.save_model(model)







    
