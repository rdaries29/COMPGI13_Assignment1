#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional Functiosn for Part 1 of Assignment
@author: russeldaries
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def weight_variables(shape):
  ini_w = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(ini_w)

def bias_variables(shape):
  ini_b = tf.constant(0.1, shape=shape)
  return tf.Variable(ini_b)

def conv_2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2by2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# Function for rendering 2D plots
def plot2D_Graph(epochs,data, xlabel, ylabel,name):
    plt.plot(epochs,data)
    plt.grid()
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend()
    plt.savefig(name)
    plt.show()

# Function for creating confusion matrix
def confusion_matrix_plot(y_true_labels, y_p, name, norm = False):

    matrix = confusion_matrix(y_true_labels, y_p)

    if norm ==True:
        matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    else:
        matrix_norm = matrix

    plt.matshow(matrix_norm)
    plt.colorbar()
    plt.ylabel('Predicated Label')
    plt.xlabel('True Label')
    plt.savefig(name)


