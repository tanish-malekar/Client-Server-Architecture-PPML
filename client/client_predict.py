import numpy as np
import h5py

import sys
import os

# Add the client directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from server_predict import getOutputLayer


def getPrediction(input):
    encryptedInput = getEncryptedInput(input)
    outputLayer = getOutputLayerFromServer(encryptedInput)
    return getResultFromOutputLayer(outputLayer)


def getEncryptedInput(input):
    return input

def getResultFromOutputLayer(outputLayer):
    #dummy:
    outputLayer=outputLayer.reshape(-1)
    predicted=np.where(outputLayer>0.5, 1, 0)
    return predicted

def getOutputLayerFromServer(encryptedInput):
    return getOutputLayer(encryptedInput)

def getActivation(Z, activation):
    if activation== "sigmoid":
        A=1/(1+np.exp(-Z))

    if activation== "relu":
        A = np.maximum(0,Z)



def main():
    #dummy:
    train_dataset = h5py.File('/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/server/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    # Select a single sample (for example, the first one)
    single_input = np.array([train_set_x_orig[0]])

    train_Xflat = single_input.reshape(-1, single_input.shape[0])

    # Scaling pixel values b/w 0 to 1
    train_X = train_Xflat /255

    print(getPrediction(train_X))

    

if __name__ == "__main__":
    main()

