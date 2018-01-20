"""
Author : Md. Mahedi Hasan
Date   : 2017-11-02
Project: cifar10_object_recognition
Description: this file is for visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import params


def plot_gallery(images, titles, n_row = 3, n_col = 3):
    plt.figure(figsize = (n_row * 2, n_col * 2))
    plt.subplots_adjust(left = 0.05,
                right = 0.95, wspace = 0.1, hspace = 0.2)

    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], size = 8)
        plt.xticks([])
        plt.yticks([])  
    plt.show()





def submit_result(result, num_test_samples):
    submit_df = pd.DataFrame({"id": range(1, num_test_samples + 1),
                              "label": result})

    submit_df.to_csv(params.RESULT_FILE_PATH + params.RESULT_FILE_NAME,
                     header = True,
                     index = False)

