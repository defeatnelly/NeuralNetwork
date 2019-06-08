#Prediction using test data
from tensorflow.keras.models import model_from_json
import numpy as np
import os
from keras.preprocessing import image

class CropsModel(object):

    CROP_LIST = ["alfalfa", "barley","corn", "soybean","wheat"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_crops(self, img):
        self.preds = self.loaded_model.predict(img)
        return CropModel.CROP_LIST[np.argmax(self.preds)]

if  __name__ == "__main__" :
    img_width, img_height = 224, 224
    base_path=os.path.join(os.getcwd(),"output","test")
    model=CropsModel("model.json","model_weights.h5")
    file=open('result.txt',"w")
    for path, subdirs, files in os.walk(base_path):
        for name in files:
            img_file= os.path.join(path, name)
            imag = image.load_img(img_file, target_size=(img_width, img_height))
            x = image.img_to_array(imag)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            pred = model.predict_crops(images)
            file.write(img_file+'  ' +img_file.split('/')[-2]+' '+pred+'\n')