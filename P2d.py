#!/usr/bin/env python3
"""
Main File for P2:d of assignment
@author: russeldaries
"""

import numpy as np
from misc_functions import *
from additional_functions import *
from part1_functions import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Activation Definitions
relu = 'ReLU'
softmax = 'softmax'
linear = 'linear'
sigmoid = 'sigmoid'

linear_deriv = 'linear_deriv'
sigmoid_deriv = 'sigmoid_deriv'
relu_deriv = 'dReLU'
softmax_deriv = 'softmax_deriv'

# Define NN sizes
epochs = 3000
batch_size = 100
eta = 0.8

# Data Input Sizes
imageSize_dim = 28
imageSize_1d = imageSize_dim * imageSize_dim 
classes = 10
hidden_layer_1 = 256
hidden_layer_1 = 256

# Storing vectors for accuracy and cross entropy achieved during training
training_accuracy_vec = []
training_cross_entropy_vec = []
epoch_vec = []

# Logic for Training and Test Mode
training_mode = False


class Neural_Network:
    
    def __init__(self,input_dim,output_dim,output_activation):
        
        if output_activation == 'LINEAR':
            self.activation = linear
            self.activation_deriv = linear_deriv
        elif output_activation == 'SIGMOID':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif output_activation == 'RELU':
            self.activation = ReLU
            self.activation_deriv = dReLU
        elif output_activation == 'SOFTMAX':
            self.activation = Softmax
            self.activation_deriv = Softmax_deriv

        sigma = 0.1
        self.b = np.zeros((1,output_dim))
        self.W = sigma*np.random.randn(input_dim,output_dim)

    #Forward Propogation in NN
    def forward_pass(self,x):
        self.layer = self.activation(self.W,x,self.b)
        return self.layer
    
    #Backward Propogation in NN
    def backward_pass(self,input_dim,output_dim,X=None):
        return self.activation_deriv(input_dim,output_dim,X,self.layer)
        
    #Updating Parameters    
    def update_Param(self, dW, db, learning_rate):
        self.W += -learning_rate * dW
        self.b += -learning_rate * db
        
#Creating instance of NN
p2d_nn_hidden1 = Neural_Network(imageSize_1d,hidden_layer_1,'RELU')
p2d_nn_hidden2 = Neural_Network(hidden_layer_1,hidden_layer_2,'RELU')
p2d_nn_output = Neural_Network(hidden_layer_2,classes,'SOFTMAX')

if(training_mode):
    print('----Training Mode Running----')
    #Ranging of epochs
    for j in range(epochs):
        
        #Sampling a batch of data
        x_BatchData,y_TrueLabels_train = mnist.train.next_batch(batch_size)
    
        # Forward Propogation
        forward_layer_hidden1 = p2d_nn_hidden1.forward_pass(x_BatchData)
        forward_layer_hidden2 = p2d_nn_hidden2.forward_pass(forward_layer_hidden1) 
        forward_layer_output = p2d_nn_output.forward_pass(forward_layer_hidden2) 
        
        #Normalising
        forward_layer_adjusted = forward_layer_output
        forward_layer_adjusted = forward_layer_adjusted - y_TrueLabels_train
        forward_layer_adjusted = forward_layer_adjusted/batch_size
        
        
        # Backward Propgation
        dW3,db3 = p2d_nn_output.backward_pass(forward_layer_hidden2.T,forward_layer_adjusted)
        dW2,db2,delta_hidden2 = p2d_nn_hidden2.backward_pass(forward_layer_adjusted,p2d_nn_output.W.T,forward_layer_hidden1.T)
        dW1,db1,delta_hidden1 = p2d_nn_hidden1.backward_pass(delta_hidden2,p2d_nn_hidden2.W.T,x_BatchData.T)
        
        #Update Parameters
        p2d_nn_hidden1.update_Param(dW1,db1,eta)
        p2d_nn_hidden2.update_Param(dW2,db2,eta)
        p2d_nn_output.update_Param(dW3,db3,eta)

        if j % 100 == 0:
            # Storing samples of training accuracy and cross-entropy
            training_accuracy,_ = Test_CostP2d(p2d_nn_hidden1,p2d_nn_hidden2,p2d_nn_output,x_BatchData,y_TrueLabels_train) 
            training_accuracy_vec.append(training_accuracy)
            epoch_vec.append(j)
                
            training_cross_entropy = crossEntropy(forward_layer_output, y_TrueLabels_train,batch_size)
            training_cross_entropy_vec.append(training_cross_entropy)
            
            print("Epochs: ",j," Training Accuracy: ",training_accuracy)
            
    y_TrueLabels_test = np.argmax(mnist.test.labels,axis=1)
    accuracy_out_test,y_PredLabels_test = Test_CostP2d(p2d_nn_hidden1,p2d_nn_hidden2,p2d_nn_output,mnist.test.images,mnist.test.labels) 
    # Result Plotting
    plot2D_Graph(epoch_vec,training_accuracy_vec, 'epochs', 'classification accuracy','q2d_train_acc.pdf')
    plot2D_Graph(epoch_vec,training_cross_entropy_vec, 'epochs', 'cross entropy loss','q2d_train_loss.pdf')
    confusion_matrix_plot(y_TrueLabels_test, y_PredLabels_test, 'q2d_confusion_matrix.pdf',norm = True)
    
    # Test Accuracy
    print("Test Accuracy: " ,accuracy_out_test)
    print("Training Accuracy: ", training_accuracy_vec[-1])
    print("Training Cross Entropy: " , training_cross_entropy_vec[-1])
    
    np.savez('model_training_weights/P2d/P2d', W1=p2d_nn_hidden1.W, b1=p2d_nn_hidden1.b ,W2=p2d_nn_hidden2.W, b2=p2d_nn_hidden2.b, W3 =p2d_nn_output.W , b3 = p2d_nn_output.b, train_acc = training_accuracy_vec[-1])
    print('Model weights saved to file: model_training_weights\P2d')
    print('----Training Mode completed and saved----')

else:
    print('----Test Mode Running----')
    saved_var = np.load('model_training_weights/P2d/P2d.npz')
    p2d_nn_hidden1.W = saved_var['W1']
    p2d_nn_hidden1.b  = saved_var['b1']
    p2d_nn_hidden2.W = saved_var['W2']
    p2d_nn_hidden2.b  = saved_var['b2']
    p2d_nn_output.W = saved_var['W3']
    p2d_nn_output.b  = saved_var['b3']
    train_acc_saved = saved_var['train_acc']

    accuracy_out_test,y_PredLabels_test = Test_CostP2d(p2d_nn_hidden1,p2d_nn_hidden2,p2d_nn_output,mnist.test.images,mnist.test.labels)
    print("Training Accuracy: " ,train_acc_saved)
    print("Test Accuracy: " ,accuracy_out_test)
    print('----Test Mode Completed----')




        