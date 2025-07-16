from ucimlrepo import fetch_ucirepo
import numpy as np
import seaborn as sns
import pandas as pd
# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

print(y.value_counts())

X = X.values.T
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = y.reshape(1,-1)

input_layer = 5
hidden_layer = 100
hidden_layer_2 = 100
output_layer = 1

#defining sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

#Defining weights and bias
def initialize_parameters(input_layer,hidden_layer,hidden_layer_2,output_layer):
    W1 = np.random.rand(hidden_layer,input_layer) * 0.01
    b1 = np.zeros((hidden_layer,1))
    W2 = np.random.rand(hidden_layer,hidden_layer_2) * 0.01
    b2 = np.zeros((hidden_layer_2,1))
    W3 = np.random.rand(output_layer,hidden_layer_2) * 0.01
    b3 = np.zeros((output_layer,1))


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


#Forward Propagation
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)

    A2 = A2.astype(float)
    A3 = A3.astype(float)

    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2,
        "Z3" : Z3,
        "A3" : A3
    }
    return A3,cache


#Compute Cost
def compute_cost(A3,y):
    m = len(y.reshape(-1,1))
    #logprobs = np.multiply(y,np.log(A3)) + np.multiply( 1-y,np.log(1-A3)  )
    cost = -np.mean(y * np.log(A3 + 1e-15) + (1 - y) * np.log(1 - A3 + 1e-15))
    cost = float(np.squeeze(cost))
    return cost


#Defining BackWard Propagation
def backward_propagation(parameters,cache,X,y):
    m = len(y.reshape(-1, 1))
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters['W3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    dz3 = A3 - y
    dw3 = np.dot(dz3,A2.T) / m
    db3 = np.sum(dz3,axis = 1, keepdims = True)

    da2 = np.dot(W3.T,dz3)
    dz2 = da2 * A2 * (1-A2)

    dw2 = np.dot(dz2, A1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    da1 = np.dot(W2.T, dz2)        # (50, m)
    dz1 = da1 * (1 - np.power(A1, 2))  # tanh'(Z1)


    dw1 = np.dot(dz1,X.T) / m
    db1= np.sum(dz1,axis = 1,keepdims = True)

    grads = {
        'dW1' : dw1,
        'dW2' : dw2,
        'dW3' : dw3,
        'db1' : db1,
        'db2' : db2,
        'db3' : db3
    }
    return grads

#Update parameters
def update_parameters(parameters,grads,learning_rate = 0.005):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    dW3 = grads['dW3']
    db3 = grads['db3']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def nn_model(X, y, input_layer, hidden_layer, hidden_layer_2, output_layer, num_iterations, print_cost=False):
    parameters = initialize_parameters(input_layer,hidden_layer,hidden_layer_2,output_layer)
    for i in range(num_iterations):
        A3,cache = forward_propagation(X,parameters)

        cost = compute_cost(A3,y)

        grads = backward_propagation(parameters,cache,X,y)

        parameters = update_parameters(parameters,grads)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))


    return parameters

params = nn_model(
    X, y,
    input_layer=30,
    hidden_layer=25,
    hidden_layer_2=25,
    output_layer=1,
    num_iterations=11000,
    print_cost=True
)

def predict(X, parameters):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)
    return predictions
preds = predict(X, params)
print("Accuracy:", np.mean(preds == y))
from ucimlrepo import fetch_ucirepo
import numpy as np
import seaborn as sns
import pandas as pd
# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

print(y.value_counts())

X = X.values.T
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
y = y.reshape(1,-1)

input_layer = 5
hidden_layer = 100
hidden_layer_2 = 100
output_layer = 1

#defining sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)

#Defining weights and bias
def initialize_parameters(input_layer,hidden_layer,hidden_layer_2,output_layer):
    W1 = np.random.rand(hidden_layer,input_layer) * 0.01
    b1 = np.zeros((hidden_layer,1))
    W2 = np.random.rand(hidden_layer,hidden_layer_2) * 0.01
    b2 = np.zeros((hidden_layer_2,1))
    W3 = np.random.rand(output_layer,hidden_layer_2) * 0.01
    b3 = np.zeros((output_layer,1))


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


#Forward Propagation
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = np.dot(W1,X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)

    A2 = A2.astype(float)
    A3 = A3.astype(float)

    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2,
        "Z3" : Z3,
        "A3" : A3
    }
    return A3,cache


#Compute Cost
def compute_cost(A3,y):
    m = len(y.reshape(-1,1))
    #logprobs = np.multiply(y,np.log(A3)) + np.multiply( 1-y,np.log(1-A3)  )
    cost = -np.mean(y * np.log(A3 + 1e-15) + (1 - y) * np.log(1 - A3 + 1e-15))
    cost = float(np.squeeze(cost))
    return cost


#Defining BackWard Propagation
def backward_propagation(parameters,cache,X,y):
    m = len(y.reshape(-1, 1))
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters['W3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    dz3 = A3 - y
    dw3 = np.dot(dz3,A2.T) / m
    db3 = np.sum(dz3,axis = 1, keepdims = True)

    da2 = np.dot(W3.T,dz3)
    dz2 = da2 * A2 * (1-A2)

    dw2 = np.dot(dz2, A1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    da1 = np.dot(W2.T, dz2)        # (50, m)
    dz1 = da1 * (1 - np.power(A1, 2))  # tanh'(Z1)


    dw1 = np.dot(dz1,X.T) / m
    db1= np.sum(dz1,axis = 1,keepdims = True)

    grads = {
        'dW1' : dw1,
        'dW2' : dw2,
        'dW3' : dw3,
        'db1' : db1,
        'db2' : db2,
        'db3' : db3
    }
    return grads

#Update parameters
def update_parameters(parameters,grads,learning_rate = 0.005):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    dW3 = grads['dW3']
    db3 = grads['db3']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def nn_model(X, y, input_layer, hidden_layer, hidden_layer_2, output_layer, num_iterations, print_cost=False):
    parameters = initialize_parameters(input_layer,hidden_layer,hidden_layer_2,output_layer)
    for i in range(num_iterations):
        A3,cache = forward_propagation(X,parameters)

        cost = compute_cost(A3,y)

        grads = backward_propagation(parameters,cache,X,y)

        parameters = update_parameters(parameters,grads)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))


    return parameters

params = nn_model(
    X, y,
    input_layer=30,
    hidden_layer=25,
    hidden_layer_2=25,
    output_layer=1,
    num_iterations=11000,
)

def predict(X, parameters):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)
    return predictions
preds = predict(X, params)
print("Accuracy:", np.mean(preds == y))
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y.flatten(), preds.flatten()))

print("\nClassification Report:")
print(classification_report(y.flatten(), preds.flatten(), target_names=['Benign', 'Malignant']))
