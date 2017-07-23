# MNIST Classifier with 1 linear layer
# Author: Russel Stuart Daries (UCABRSD)
# Import all nesscary packages

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from part1_functions import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Size Declarations
batch_size = 100
image_vec_1d = 784 #28*28
output_classes = 10  #0 to 9 
epochs = 10000
learning_rate = 0.8
training_mode = True


# Storing vectors for accuracy and cross entropy achieved during training
training_accuracy_vec = []
training_cross_entropy_vec = []
epoch_vec = []

# Declarations
x =  tf.placeholder(tf.float32,[None, image_vec_1d])
W =  tf.Variable(tf.zeros([image_vec_1d,output_classes]))
b =  tf.Variable(tf.zeros([output_classes]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, output_classes])

# Defining Cross-Entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Raw formula for cross-entropy-numerically unstable
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Prediction 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#Accuracy calculation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saving variable for TensorFlow model
saver = tf.train.Saver()

# Initializing all variables

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Random sampling of batches of 100 samples (Stochastic Graident Descent)
            # Confusion Matrix of results
    y_true_labels = np.argmax(mnist.test.labels, 1)
    y_pred_labels = tf.argmax(y, 1)   
    if(training_mode):
        print('----Training Mode Running----')
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
        plot2D_Graph(epoch_vec,training_accuracy_vec, 'epochs', 'classification accuracy','q1a_train_acc.pdf')
        plot2D_Graph(epoch_vec,training_cross_entropy_vec, 'epochs', 'cross entropy loss','q1a_train_loss.pdf')
    
        accuracy_out,y_p = sess.run([accuracy,y_pred_labels], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    
        confusion_matrix_plot(y_true_labels, y_p, 'q1a_confusion_matrix.pdf',norm = True)
        
    #    confusion_matrix_plot(y_true_labels, y_p, True,)
        
        # Saving the model
        save_path = saver.save(sess,'model_training_weights/P1a/tfP1a.ckpt')
        print('Model saved to file: ', save_path)
        
        #Final computation
        print("Test Accuracy: " ,accuracy_out)
        print("Test Cross Entropy: " ,sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        print("Training Accuracy: ", training_accuracy_vec[-1])
        print("Training Cross Entropy: " , training_cross_entropy_vec[-1])
        print('----Training Mode completed and saved----')

    else:
        print('----Test Mode Running----')
        saver.restore(sess, "model_training_weights/P1a/tfP1a.ckpt")
        
        accuracy_out_restored,y_p = sess.run([accuracy,y_pred_labels], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Test Accuracy: " ,accuracy_out_restored)
        print('----Test Mode Completed----')
        
    