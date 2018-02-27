#Thong Tran

# Libraries
import NN
import numpy as np
import pandas as pd

# MNIST CSV Dataset will be the inputs
CSV_INPUTS = {'train_set': "mnist_train.csv", 'test_set': "mnist_test.csv"}
# Learning Rates for project
RATES = (0.1, 0.01, 0.001)
# Train the network for 50 epochs
EPOCH = 50

if __name__ == "__main__":
    # Setup the inputs
    train_input = pd.read_csv(CSV_INPUTS['train_set'], sep=',', engine='c', header=None, na_filter=False,
                                dtype=np.float64,
                                low_memory=False).as_matrix()

    test_input = pd.read_csv(CSV_INPUTS['test_set'], sep=',', engine='c', header=None, na_filter=False,
                               dtype=np.float64,
                               low_memory=False).as_matrix()

    # Preprocessing
    for x in train_input:
        x[1:] /= 255
    for x in test_input:
        x[1:] /= 255

    # hidden and output unit has a weighted connection, whose value is set to 1
    train_input = np.insert(train_input, 1, [1], axis=1)
    test_input = np.insert(test_input, 1, [1], axis=1)

    # Experiment 1: Vary number of hidden units
    print("Starting Experiment 1")
    for n in [20, 50, 100]:
        print("With n = ", n)
        p_cluster = NN.NeuralNetwork(n, 10, 0.9, 0.1)

        for x in range(EPOCH):
            # Calculate Test and Training on repeat
            test_accuracy = p_cluster.run_epoch(test_input, False, False)
            train_accuracy = p_cluster.run_epoch(train_input, True, False)
            print('Epoch', x, '  Training Accuracy: ', train_accuracy, '  Testing Accuracy: ', test_accuracy)

        p_cluster.run_epoch(test_input, False, True)
        print(p_cluster.confusion_matrix)

    #Experiment 2: Vary the momentum value
    print("Starting Experiment 2")
    for momentum in [0, 0.25, 0.5]:
        print("momentum values = ", momentum)
        p_cluster = NN.NeuralNetwork(100, 10, momentum, 0.1)

        for x in range(EPOCH):
            # Calculate Test and Training on repeat
            test_accuracy = p_cluster.run_epoch(test_input, False, False)
            train_accuracy = p_cluster.run_epoch(train_input, True, False)
            print('Epoch', x, '  Training Accuracy: ', train_accuracy, '  Testing Accuracy: ', test_accuracy)

        p_cluster.run_epoch(test_input, False, True)
        print(p_cluster.confusion_matrix)

    #Experiment 3: Vary the number of training examples
    print("Starting Experiment 3")
    for case in [2, 4]:
        print("Case: ", case)
        p_cluster = NN.NeuralNetwork(100, 10, 0.9, 0.1)
        partial_training_data = np.array(train_input[::case])
        for x in range(EPOCH):
            # Calculate Test and Training on repeat
            test_accuracy = p_cluster.run_epoch(test_input, False, False)
            train_accuracy = p_cluster.run_epoch(partial_training_data, True, False)
            print('Epoch', x, '  Training Accuracy: ', train_accuracy, '  Testing Accuracy: ', test_accuracy)

        p_cluster.run_epoch(test_input, False, True)
        print(p_cluster.confusion_matrix)
