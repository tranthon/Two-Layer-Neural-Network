# Two-Layer-Neural-Network
Implementation of a two-layer network(i.e, one hidden-layer) to perform the handwritten digit recognition task og the MNIST dataset

Experiment 1: Vary number of hidden units

Description: This experiment was tested with n = 20, 50, and 100. Then, the program trains on the training set for each value of n. After each training example, the weights were changed. After each epoch, the program calculates the network's accuracy on the training set and the test set. The program was tested with 50 epochs. After the training is complete, the program creates a confusion matrix for each of the trained networks.

Experiment 2: Vary the momentum value

Description: the number of hidden units was fixed to 100, and the momentum value was varied during training. The program uses momentum values of 0, 0.25, and 0.5. The program creates a confusion matrix for each of the trained networks. I plot the results as in the 1st experiment.

Experiment 3: Vary the number of training examples

Description: the number of hidden units was fixed to 100 and momentum 0.9. The program trains two networks, using respectively one quarter and one half of the training examples for training. Plot the results, as in the previous experiments, plotting accuracy on both the training and test data at the end of each epoch. The program then creates a confusion matrix for each of the trained networks
