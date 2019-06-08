#Quick data visualization
# display some images for every different crops

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import os
import cv2

train=os.path.join(os.getcwd(),"output","train/")
# input path for the images
def data_visualize(train,pic_size = 224):
    plt.figure(0, figsize=(12,20))
    cpt = 0
    for crops in os.listdir(train):
        for i in range(1,6):
            cpt = cpt + 1
            plt.subplot(7,5,cpt)
            img = load_img(train + crops + "/" +os.listdir(train + crops)[i], target_size=(pic_size, pic_size))
            plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.show()

data_visualize(train,224)

def number_of_images(train):
    # count number of train images for each crop
    for crop in os.listdir(train):
        print(str(len(os.listdir(train + crop))) + " " + crop + " images")

number_of_images(train)

