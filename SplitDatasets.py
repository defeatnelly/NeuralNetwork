#Split to training, validation and test dataset
#params: inputpath,outputpath, seed to reproduce
#param: ratio default for training and validation (.8, .2) for ML and 
# training, validation and test for NN
inputpath=input("Enter the data folder: ")

def split_folders_ML(inputpath="./Imagery",outputpath="output",seed=1337,ratio=(.8, .2)):
    try:
        split_folders.ratio(inputpath, outputpath, seed, ratio) # default values
        if len(ratio)==3:
            test,train,validation=[folder for folder in os.listdir(outputpath)]
            train,validation=os.path.abspath(outputpath)+"/"+train+"/", os.path.abspath(outputpath)+"/"+validation+"/"
            test=os.path.abspath(outputpath)+"/"+test+"/"
            print("Location for the training, validation and test sets: ",  train,validation,test)
        else:
            train,test=[folder for folder in os.listdir(outputpath)]
            train,validation=os.path.abspath(outputpath)+"/"+train+"/", os.path.abspath(outputpath)+"/"+validation+"/"
            print("Location for the training and validation sets: ",  train,validation)
    except:
        print('Check the entered value:\n Input path: {0}\n, Output path:{1}\n,seed: {2}\n, ratio (tuple ranges 0 to 1): {3}\n',
              inputpath,outputpath,seed,ratio)
    return train,validation,test

#
train,validation,test=split_folders_ML(inputpath,"output",1337,(.8,.1,.1))