# using keras ImageDataGenerator to perform data augmentation such as (randomly rotating the image, zooming, etc.)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# batch_size number of images to feed into the NN for every batch

base_path = os.path.join(os.getcwd(),"output/")
def image_Data_Generator(batch_size = 20,pic_size = 224):
    datagen_train = ImageDataGenerator()
    datagen_validation = ImageDataGenerator()


    train_generator = datagen_train.flow_from_directory(base_path + "train",
                                                        target_size=(pic_size,pic_size),
                                                        color_mode="rgb",
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = datagen_validation.flow_from_directory(base_path + "val",
                                                                  target_size=(pic_size,pic_size),
                                                                  color_mode="rgb",
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  shuffle=False)
    return train_generator,validation_generator

train_generator,validation_generator=image_Data_Generator(20,224) 