#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to store all activation functions required in neural network
@author: russeldaries
"""
import numpy as np

#def activation_layer(w,x,b,activation):
#    if activation == 'linear':
#        z = (np.dot(w,x)+b)
#    elif activation == 'sigmoid':
#        z = sigmoid(np.dot(w,x)+b)
#    elif activation == 'relu':
#        z = relu(np.dot(w,x)+b)
#    elif activation == 'softmax':
#        z = softmax(np.dot(w,x)+b)
#    return z

# Sigmoid Function
def sigmoid(z):
    y = sigmoid(1/1+(np.exp(-z)))
    return y

# Sigmoid Derivative
def sigmoid_deriv(z):
    dz = sigmoid(z)*(1-sigmoid(z))
    return dz

# ReLU Function
def ReLU(W,X,b):
    Z = np.dot(X,W) + b
    y = np.maximum(0,Z)
    return y

# ReLU Derivative
def dReLU(output,W,X,hidden_layer):

    delta_hidden = np.dot(output, W)
    delta_hidden[hidden_layer <= 0] = 0 # Gradient of 0
    #delta_hidden[delta_hidden >  0] = 1 # Gradient of 1

    delta_W = np.dot(X,delta_hidden)
    delta_b = np.sum(delta_hidden,axis = 0 ,keepdims = True)

    return (delta_W,delta_b,delta_hidden)

# Softmax Function
def Softmax(W,X,b):
    z  = np.dot(X,W) + b
    y = np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)
    return y

# Softmax Derivative
def Softmax_deriv(hidden_layer,output,x,h):
    delta_W = np.dot(hidden_layer,output)
    delta_b = np.sum(output,axis= 0,keepdims = True)
    return (delta_W,delta_b)

# Cross - Entropy Function
def crossEntropy(x,y,batch_size):
    loss = -np.log(np.sum(x*y,axis=1))
    return np.sum(loss)/batch_size

# Function to calculate accuracy
def Test_Cost(nn_layers,X_test,y_test):

    output_prob = Softmax(nn_layers.W,X_test,nn_layers.b)
    y_PredLabel = np.argmax(output_prob,axis = 1)
    y_TrueLabel = np.argmax(y_test,axis=1)
    test_accuracy = np.mean(y_PredLabel == y_TrueLabel)

    return (test_accuracy,y_PredLabel)

# Function to calculate accuracy for P2:d
def Test_CostP2c(hidden_layer_1,output_layer,X_test,y_test):

    hidden_layer_output = np.maximum(0, np.dot(X_test, hidden_layer_1.W) + hidden_layer_1.b)
    output_prob = Softmax(output_layer.W,hidden_layer_output,output_layer.b)
    y_PredLabel = np.argmax(output_prob,axis = 1)
    y_TrueLabel = np.argmax(y_test,axis=1)
    test_accuracy = np.mean(y_PredLabel == y_TrueLabel)

    return (test_accuracy,y_PredLabel)

# Function to calculate accuracy for P2:d
def Test_CostP2d(hidden_layer_1,hidden_layer_2,output_layer,X_test,y_test):

    hidden_layer_output1 = np.maximum(0, np.dot(X_test, hidden_layer_1.W) + hidden_layer_1.b)
    hidden_layer_output2 = np.maximum(0, np.dot(hidden_layer_output1, hidden_layer_2.W) + hidden_layer_2.b)
    output_prob = Softmax(output_layer.W,hidden_layer_output2,output_layer.b)
    y_PredLabel = np.argmax(output_prob,axis = 1)
    y_TrueLabel = np.argmax(y_test,axis=1)
    test_accuracy = np.mean(y_PredLabel == y_TrueLabel)

    return (test_accuracy,y_PredLabel)


