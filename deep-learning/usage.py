from .utils import load_data
from .neural_network import initialize_parameters, forward_propagation, compute_cost, backward_propagation, update_parameters


X_train, X_test, Y_train, Y_test = load_data()
layer_dims = [len(X_train), 20, 7, 5, 1]
learning_rate = 0.01

X = X_train
Y = Y_train

parameters = initialize_parameters(layer_dims)
AL, caches = forward_propagation(X, parameters)
cost = compute_cost(AL, Y)
grads = backward_propagation(AL, Y, caches)
parameters = update_parameters(parameters, grads, learning_rate)