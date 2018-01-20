"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
Description : this file is for test dataset
"""

import numpy as np

import image_preprocessing
import model_utils
import params
import vis_utils


model_path_list = params.get_model_path_list(nb_model = 1)
model_weight_list = params.get_model_weight_list(nb_model = 1)
custom_obj = [None, None]


#setting parameters
batch_size = 128
nb_test = params.NUM_TEST_SAMPLES
X_test = []    
predictions_1 = []
total_img = 0


print("Loading Model 1")
model_1 = model_utils.read_model(model_path_list[0],
                               model_weight_list[0],
                               custom_obj[0])
   

print("Loading test images....")
test_data = image_preprocessing.restore_data(params.TEST_DATA)
print("Test samples: ", test_data.shape)



for i in range(0, params.NUM_TEST_SAMPLES):
    img = test_data[i]

    X_test.append(image_preprocessing.preprocess_input(img))
    total_img += 1

    if ((total_img % 5000 == 0) or (total_img == params.NUM_TEST_SAMPLES)):
        print("reading {} images".format(i + 1))

        data = np.array(X_test, dtype = np.float32)
        X_test.clear()

        print("predicting 5000 images for model 1")
        pred = model_1.predict(data, batch_size, verbose = 2)
        predictions_1 = predictions_1 + pred.tolist()

del test_data, model_1



# finding class value
y_label = np.argmax(predictions_1, axis = 1)
result = [params.CLS_NAMES[i] for i in y_label]



# result
print("preparing for result submission")
vis_utils.submit_result(result, params.NUM_TEST_SAMPLES)















