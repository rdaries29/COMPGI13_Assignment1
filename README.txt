Created on Tue Feb  7 19:02:55 2017
README.txt file with instructions for running
@author: russeldaries (UCABRSD) - University College London
COMPGI13 Assignment 1
----------------------------
CONFIGURATION INSTRUCTIONS
----------------------------
This software structure for various configurations of Neural Netwok for COMPGI13: Advanced Topics in Machine Learning was developed
and tested

----------------------------
INSTALLATION INSTRUCTIONS
----------------------------
The source code provided was written in the Python Programming Language thus there are a few key packages required to run this code:
    - Python 3.5
    - TensorFlow v0.12
    - Scikit-Learn (sklearn.metrics package)
    - Numpy 1.11.0
    - Matplotlib (matplotlib.pyplot)
    - MNIST Handwritten Dataset (contained within TensorFlow download)
----------------------------
FILE MANIFEST
----------------------------
The source code comes in the following hierarchy:
-UCABRSD.pdf: This is the assignment report as required.
    - model_training_weights (Directory containing all *.ckpt  and *.npz files for restoring trainined neural network weights)

-P1a.py: Source Code for question P1a of assignment.
-P1b.py: Source Code for question P1b of assignment.
-P1c.py: Source Code for question P1c of assignment.
-P1d.py: Source Code for question P1d of assignment.

-P2b.py: Source Code for question P2b of assignment.
-P2c.py: Source Code for question P2c of assignment.
-P2d.py: Source Code for question P2d of assignment.

-misc_functions.py: Misschelantius functions for building NN architectures.
-part1_functions.py: File for storing plotting and confusion matrix function declarations.
-additional_functions.py: File for storing 1-hot vector mapping of classifier.
----------------------------
OPERATING INSTRUCTIONS
----------------------------
Each file titled P*.py corresponds to a question in the assignment. Each of these files has a boolen statement:
    -   training_mode = False 
This statement in its current form when executing P*.py will place the script in Testing mode and will load the saved model
weights and test the NN on the test set and output a test score through a print statement.

When training_mode = True, the script goes into training mode and performs Stochastic Gradient Descent on the MNIST dataset in order
to calculate the NN weights saving it a  *.ckpt or *.npz file. In each update step the training accuracy is printed to the console as well as the final test accuracy of the model.

--------------------------------------
COPYRIGHT AND LICENSING INFORMATION
--------------------------------------
All the containing source code is copyrighted and not distributable to anyone without prior conscent by the author of this material.
----------------------------------
CONTACT INFORMATION OF PROGRAMMER
----------------------------------
UCL ID: UCABRSD
Email: russel.daries.16@ucl.ac.uk
----------------------------------
CREDITS AND ACKNOWLEDGEMENTS
----------------------------------
Fundamental knowledge of Neural Networks was provided through the lecture material and lecturers of Google DeepMind and University College London
for the course COMPGI13: Advanced Topics in Machine Learning.
