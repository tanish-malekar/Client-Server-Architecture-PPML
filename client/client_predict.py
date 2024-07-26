import numpy as np
import h5py
from Pyfhel import Pyfhel

import sys
import os
import base64

# Add the client directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../encryption')))

from server_predict import getOutputLayer
from encryption import decryptMatrix, multipleCiphers, encryptMatrix, initializeServer


def initialize_HE():
    n_mults = 10
    ckks_params = {
        'scheme': 'CKKS',
        'n': 2**14,
        'scale': 2**31,
        'qi_sizes': [60] + [30]*n_mults + [60]
    }
    HE = Pyfhel()
    HE.contextGen(**ckks_params)
    HE.keyGen()
    HE.relinKeyGen()
    return HE

#global in client
HE = initialize_HE()

def initiateConnection():
    publicKey = HE.to_bytes_public_key(compr_mode=u'zstd')
    publickKey64 = base64.b64encode(publicKey).decode('utf-8')
    context = HE.to_bytes_context(compr_mode='zstd')
    context64 = base64.b64encode(context).decode('utf-8')
    relinKey = HE.to_bytes_relin_key(compr_mode='zstd')
    relinKey64 = base64.b64encode(relinKey).decode('utf-8')

    sendInitializationToServer(publickKey64, context64, relinKey64)


def sendInitializationToServer(publickKey64, context64, relinKey64):
    initializeServer(publickKey64, context64, relinKey64)


def getPrediction(input):
    encryptedInput = getEncryptedInput(input)
    outputLayer = getOutputLayerFromServer(encryptedInput)
    return getResultFromOutputLayer(outputLayer)


def getEncryptedInput(input):
    # encryptedInput = encryptMatrix(input)
    # return encryptedInput
    return input

def getResultFromOutputLayer(outputLayer):
    # decryptedOutputLayer = decryptMatrix(outputLayer)
    # decryptedOutputLayer=decryptedOutputLayer.reshape(-1)
    # predicted=np.where(decryptedOutputLayer>0.5, 1, 0)
    # return predicted
    outputLayer=outputLayer.reshape(-1)
    predicted=np.where(outputLayer>0.5, 1, 0)
    return predicted

def getOutputLayerFromServer(encryptedInput):
    # TODO: Call server to get output layer
    return getOutputLayer(encryptedInput)

def getActivation(Z, activation):
    if activation== "sigmoid":
        A=1/(1+np.exp(-Z))

    if activation== "relu":
        A = np.maximum(0,Z)



def main():
    #dummy:
    train_dataset = h5py.File('/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    # Select a single sample (for example, the first one)
    single_input = np.array([train_set_x_orig[0]])

    train_Xflat = single_input.reshape(-1, single_input.shape[0])

    # Scaling pixel values b/w 0 to 1
    train_X = train_Xflat /255

    print(getPrediction(train_X))

    

if __name__ == "__main__":
    main()

