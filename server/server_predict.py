import pickle
import numpy as np
import time

import sqlite3
import json
from Pyfhel import Pyfhel
import base64
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../client')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../encryption')))

from shared_functions import getActivation
from encryption import decryptMatrix, multipleCiphers, encryptMatrix

layerDims=[10, 47, 62, 67, 72, 62, 1]
activationFunctions = ["tanh", "tanh", "tanh", "sigmoid", "relu", "sigmoid"]

def getOutputLayer(encryptedInput):
    encryptedParams = getEncryptedParameters()
    outputLayer = l_LayerForwardEncrypted(encryptedInput, encryptedParams, layerDims)
    return outputLayer

def getEncryptedParameters():
    # conn = sqlite3.connect('ppml.db')
    # cursor = conn.cursor()
    # cursor.execute('''
    #     SELECT encrypted_parameters FROM user_details WHERE id = 1
    # ''')
    # encryptedParameters = json.loads(cursor.fetchone()[0])
    # return encryptedParameters

    print("reading encrypted parameters...")
    paramaters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/client/encrypted_parameters.pkl", 'rb'))    
    print(type(paramaters))
    print(paramaters.keys())
    return paramaters

    # paramaters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/deep-learning/model_details.pkl", 'rb'))
    # return paramaters

def encryptParameters(parameters, HE):
    print("encrypting parameters...")
    encryptedParameters = {}
    for key, value in parameters.items():
        encryptedParameters[key] = encryptMatrix(value, HE)
    print("encryption done.")
    return encryptedParameters


def l_LayerForwardEncrypted(encryptedInput, encryptedParams, layerdims):
    L =  len(layerdims)-1
    A = encryptedInput

    for l in range(1,L+1):
        A_prev = A
        A = forward(A_prev, encryptedParams["W"+str(l)], encryptedParams["b"+str(l)], activationFunctions[l-1])

    return A


def forward(A_prev, W, b, activation):
    
 
    Z = np.add( np.dot(A_prev, W), b)


    A = calculateActivationFunction(Z, activation)

    return A


def add_random_noise(Z):
    noise_indices = []
    
    # Flatten Z and compute min and max values
    Z_flat = Z.flatten()
    min_val, max_val = Z_flat.min(), Z_flat.max()
    
    Z_noisy = []
    for i in range(len(Z)):
        if np.random.rand() > 0.5:  # Randomly decide whether to add noise
            noise = np.random.uniform(min_val, max_val, 1)  # Generate random value within the range
            Z_noisy.append(noise)  # Append noise
            noise_indices.append(len(Z_noisy))  # Record index where noise was added
        Z_noisy.append(Z[i])
    
    return np.array(Z_noisy), noise_indices

def remove_noise(Z, noise_indices):
    Z_cleaned = np.delete(Z, noise_indices, 0)
    return Z_cleaned


def calculateActivationFunction(Z, activation):
    # Z_noisy, noise_indices = add_random_noise(Z)
    # activationFromClient= getActivationFromClient(Z_noisy, activation)
    # activationFromClient_cleaned = remove_noise(activationFromClient, noise_indices)
    # return activationFromClient_cleaned
    return getActivationFromClient(Z, activation)


def getActivationFromClient(Z, activation):
    # TODO: Call client to get activation
    return getActivation(Z, activation)


def initializeServer(publickKey64, context64, relinKey64):
    print("hi from server")
    # conn = sqlite3.connect('ppml.db')
    # cursor = conn.cursor()
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS user_details
    #     (id INTEGER PRIMARY KEY,
    #     encrypted_parameters TEXT)
    # ''')

    HE = Pyfhel()
    HE.from_bytes_context(base64.b64decode(context64))
    HE.from_bytes_public_key(base64.b64decode(publickKey64))
    HE.from_bytes_relin_key(base64.b64decode(relinKey64))

    paramaters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/deep-learning/model_details.pkl", 'rb'))
    curr_time = time.time()
    encryptedParameters = encryptParameters(paramaters, HE)
    print("Encryption time: ", time.time() - curr_time)
    with open('encrypted_parameters.pkl', 'wb') as file:
        pickle.dump(encryptedParameters, file)

   
    # encryptedParametersJSON = json.dumps(encryptedParameters)
    
    # cursor.execute('''
    #     INSERT INTO user_details (id, encrypted_parameters)
    #     VALUES (?, ?)
    # ''', (1, encryptedParametersJSON ))