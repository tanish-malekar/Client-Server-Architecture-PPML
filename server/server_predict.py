import pickle
import numpy as np
from encryption import decryptMatrix, multipleCiphers, encryptMatrix
import sqlite3


layerDims=[12288, 20, 7, 5, 1]

def getOutputLayer(encryptedInput):
    encryptedParams = getEncryptedParameters()
    outputLayer = l_LayerForwardEncrypted(encryptedInput, encryptedParams, layerDims)
    return outputLayer

def getEncryptedParameters():
    # TODO: Load encrypted parameter from db
    paramaters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/parameters.pkl", 'rb'))
    
    # to test without db: 
    # encryptedParameters = encryptParameters(parameters)
    # return encryptedParameters


    return paramaters

def encryptParameters(parameters):
    encryptedParameters = {}
    for key, value in parameters.items():
        encryptedParameters[key] = encryptMatrix(value)
    return encryptedParameters


def l_LayerForwardEncrypted(encryptedInput, encryptedParams, layerdims):
    ''' Forward propagation for L-layer

    [LINEAR -> RELU]*(L-1)   ->    LINEAR->SIGMOID

    X: Input matrix (input size/no. of features, no. of examples/BatchSize)
    parameters: dict of {W1,b1 ,W2,b2, ...}
    layerdims: Vector, no. of units in each layer  (no. of layers,)

    Returns:
    y_hat: Output of Forward Propagation
    caches: (A_prev,W,b,Z) *(L-1 times , of 1,2,..L layers)
    '''

    L =  len(layerdims)-1
    A = encryptedInput


    # L[0] is units for Input layer
    # [LINEAR -> RELU]*(L-1)    Forward for L-1 layers
    for l in range(1,L):
        A_prev = A
        A = forward(A_prev, encryptedParams["W"+str(l)], encryptedParams["b"+str(l)], "relu")
    

    # Forward for Last layer
    # [Linear -> sigmoid]
    outputLayer =forward(A, encryptedParams["W"+str(l+1)], encryptedParams["b"+str(l+1)], "sigmoid")
  

    return outputLayer


def forward(A_prev, W, b, activation):
    ''' Forward Propagation for Single layer
    A_prev: Activation from previous layer (size of previous layer, Batch_size)
        A[0] = X
    W: Weight matrix (size of current layer, size of previous layer)
    b: bias vector, (size of current layer, 1)

    Returns:
    A: Output of Single layer
    cache = (A_prev,W,b,Z), these will be used while backpropagation
    '''
    # Linear
    Z = np.add( np.dot(W,A_prev), b)

    # Activation Function
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
    Z_noisy, noise_indices = add_random_noise(Z)
    activationFromClient= getActivationFromClient(Z_noisy, activation)
    activationFromClient_cleaned = remove_noise(activationFromClient, noise_indices)
    return activationFromClient_cleaned

def getActivationFromClient(Z, activation):
    # TODO: Call client to get activation
    return getActivation(Z, activation)


#should be in client
def getActivation(Z, activation):
    if activation== "sigmoid":
        A=1/(1+np.exp(-Z))

    if activation== "relu":
        A = np.maximum(0,Z)

    return A


def initializeServer(publickKey64, context64, relinKey64):
    

    # conn = sqlite3.connect('ppml.db')
    # cursor = conn.cursor()
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS user_para
    #     (id INTEGER PRIMARY KEY,
    #     parameters TEXT)
    # ''')

    # paramaters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/parameters.pkl", 'rb'))
    # encryptedParameters = encryptParameters(paramaters)
    # # TODO to complete



    


