# MNIST Classifier with 1 hidden layer
# Author: Russel Stuart Daries (UCABRSD)
# Import all nesscary packages

import tensorflow as tf
import numpy as np
from part1_functions import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Size Declarations
batch_size = 100
image_vec_1d = 784 #28*28
output_classes = 10  #0 to 9 
epochs = 3000
learning_rate = 0.8
hidden_layer_1 = 128
training_mode = False

# Storing vectors for accuracy and cross entropy achieved during training
training_accuracy_vec = []
training_cross_entropy_vec = []
epoch_vec = []

# Defining weight variable function

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Defining bias variable function

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Defining placeholders
x = tf.placeholder(tf.float32, shape=[None, image_vec_1d])
y_ = tf.placeholder(tf.float32, shape=[None, output_classes])

# Defining Variables

# Layer 1
W1 = weight_variable([image_vec_1d,hidden_layer_1])
b1 = bias_variable([hidden_layer_1])
h1 = tf.nn.relu(tf.matmul(x,W1)+b1)

# Layer 2
W2 = weight_variable([hidden_layer_1,output_classes])
b2 = bias_variable([output_classes])

# Dropout applied to reduce overfitting
#keep_prob = tf.placeholder(tf.float32)
#h1_drop = tf.nn.dropout(h1, keep_prob)

# Output layer with dropout
#y = tf.matmul(h1_drop,W2) + b2

#Output layer without dropout
y = tf.matmul(h1,W2) + b2

# Cross entropy calculation
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# Standard Gradient Descent Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Adam Optimizer
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#Accuracy calculation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Model Saver
saver = tf.train.Saver()

# Initializing all variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_true_labels = np.argmax(mnist.test.labels, 1)
    y_pred_labels = tf.argmax(y, 1) 
    if(training_mode):
        print('----Training Mode Running----')
        # Random sampling of batches of 100 samples (Stochastic Graident Descent)
        for i in range(epochs):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            
            if i%100 == 0:
                
                training_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
                training_accuracy_vec.append(training_accuracy)
                epoch_vec.append(i)
                
                training_cross_entropy = cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys})
                training_cross_entropy_vec.append(training_cross_entropy)
    
        # Training Plots
        plot2D_Graph(epoch_vec,training_accuracy_vec, 'epochs', 'classification accuracy','q1b_train_acc.pdf')
        plot2D_Graph(epoch_vec,training_cross_entropy_vec, 'epochs', 'cross entropy loss','q1b_train_loss.pdf')
    
        # Confusion Matrix of results
      
        
        accuracy_out,y_p = sess.run([accuracy,y_pred_labels], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    
        confusion_matrix_plot(y_true_labels, y_p, 'q1b_confusion_matrix.pdf',norm = True)
        
    #    confusion_matrix_plot(y_true_labels, y_p, True,)
        
        # Saving the model
        save_path = saver.save(sess,"model_training_weights/P1b/tfP1b.ckpt")
        print('Model saved to file: ', save_path)
        
        #Final computation
        print("Test Accuracy: " ,accuracy_out)
        print("Test Cross Entropy: " ,sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Training Accuracy: ", training_accuracy_vec[-1])
        print("Training Cross Entropy: " , training_cross_entropy_vec[-1])
        print('----Training Mode completed and saved----')

    else:
        print('----Test Mode Running----')
        saver.restore(sess, "model_training_weights/P1b/tfP1b.ckpt")
        accuracy_out_restored,y_p = sess.run([accuracy,y_pred_labels], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test Accuracy: " ,accuracy_out_restored)
        print('----Test Mode Completed----')
        
