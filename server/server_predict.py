import pickle;
import numpy as np;


layerDims=[12288, 20, 7, 5, 1]

def getOutputLayer(encryptedInput):
    encryptedParams = getEncryptedParameters()
    outputLayer = l_LayerForwardEncrypted(encryptedInput, encryptedParams, layerDims)
    return outputLayer

def getEncryptedParameters():
    #fetch encryptedParams from DB and return
    #dummy:
    parameters=pickle.load(open("/Users/tanishmalekar/Library/CloudStorage/OneDrive-BajajFinanceLimited/Desktop/Research Paper/parameters.pkl", 'rb'))
    return parameters

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

    cache=(A_prev,W,b,Z)

    return A, cache


def calculateActivationFunction(Z, activation):
    #add noise to layer
    activationFromClient= getActivationFromClient(Z, activation)
    #remove noise
    return activationFromClient

def getActivationFromClient(Z, activation):
    #call client
    #dummy:
    return getActivation(Z, activation)


#should be in client
def getActivation(Z, activation):
    if activation== "sigmoid":
        A=1/(1+np.exp(-Z))

    if activation== "relu":
        A = np.maximum(0,Z)




