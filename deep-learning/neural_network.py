import numpy as np

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1  # Number of layers excluding input layer

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def forward_propagation(X, parameters):
    A = X
    caches = []
    L = len(parameters) // 2  # Number of layers

    for l in range(1, L):
        Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]
        A = np.maximum(0, Z)  # ReLU activation function
        cache = (Z, A)
        caches.append(cache)

    Z = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = 1 / (1 + np.exp(-Z))  # Sigmoid activation function
    cache = (Z, AL)
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]  # Number of training examples

    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m

    return cost

def backward_propagation(AL, Y, caches, parameters):
    grads = {}
    L = len(caches)  # Number of layers
    m = AL.shape[1]  # Number of training examples
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    dZ = dAL * AL * (1 - AL)
    grads['dW' + str(L)] = np.dot(dZ, caches[L - 1][1].T) / m
    grads['db' + str(L)] = np.sum(dZ, axis=1, keepdims=True) / m

    for l in reversed(range(L - 1)):
        dA = np.dot(parameters['W' + str(l + 2)].T, dZ)
        dZ = np.multiply(dA, np.int64(caches[l][1] > 0))
        grads['dW' + str(l + 1)] = np.dot(dZ, caches[l][0].T) / m
        grads['db' + str(l + 1)] = np.sum(dZ, axis=1, keepdims=True) / m

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers

    for l in range(L):
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters