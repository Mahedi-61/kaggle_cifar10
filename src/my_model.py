"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
"""

import numpy as np

from keras.models import Model
from keras.layers import (Conv2D, Activation, MaxPooling2D,
                          AveragePooling2D, Flatten, Dense, Input,
                          BatchNormalization, add)

from keras.regularizers import l2


def my_resent_18(input_shape, nb_classes):
    #constructing model
    nb_filters = 64
    use_max_pool = False
    nb_blocks = 4
    nb_sub_blocks = 2

    inputs = Input(shape = input_shape)
    x = Conv2D(nb_filters,
               kernel_size = 7,
               padding = "same",
               strides = 2,
               kernel_initializer = "he_normal",
               kernel_regularizer = l2(1e-4))(inputs)

    x = BatchNormalization()(x)
    x = Activation("relu") (x)

    #cifar-10 images are too small to be maxpooled. so skip it
    if use_max_pool:
        x = MaxPooling2D(pool_size = 3, strides = 2, padding="same")(x)
        nb_blocks = 3

        
    #state of block
    for i in range(nb_blocks):
        for j in range(nb_sub_blocks):
            strides = 1
            
            is_first_layer_but_not_first_block = (j == 0 and i > 0)
            if is_first_layer_but_not_first_block:  
                strides = 2

            y =  Conv2D(nb_filters,
                   kernel_size = 3,
                   padding = "same",
                   strides = strides,
                   kernel_initializer = "he_normal",
                   kernel_regularizer = l2(1e-4))(x)
            y = BatchNormalization()(y)
            y = Activation("relu")(y)


            y =  Conv2D(nb_filters,
                   kernel_size = 3,
                   padding = "same",
                   kernel_initializer = "he_normal",
                   kernel_regularizer = l2(1e-4))(y)
            y = BatchNormalization()(y)


            if is_first_layer_but_not_first_block:
                x =  Conv2D(nb_filters,
                       kernel_size = 1,
                       padding = "same",
                       strides = 2,
                       kernel_initializer = "he_normal",
                       kernel_regularizer = l2(1e-4))(x)

            x = add([x, y])
            x = Activation("relu")(x)

        nb_filters = nb_filters * 2


    #Add classifier on top
    x = AveragePooling2D()(x) #default pool_size=(2, 2)
    y = Flatten()(x)
    y = Dense(nb_classes,
              activation = "softmax",
              kernel_initializer = "he_normal")(y)

    return Model(inputs = inputs, outputs = y)









