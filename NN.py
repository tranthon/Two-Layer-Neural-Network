# Thong Tran

import numpy as np

NN_INPUT = 784


def sig_func(x):
    return 1 / (1 + np.exp(-1.0 * x))


class NeuralNetwork:
    def __init__(self, hidden_node, output_node, momentum, rate):

        # Intial setup for variables
        self.confusion_matrix = np.zeros((10, 10), int)
        self.targets = np.zeros((10, 10), float)
        self.targets.fill(0.1)
        for x in range(10):
            self.targets[x, x] = 0.9
        # Number of nodes for hidden and output
        self.hidden_node = hidden_node
        self.output_node = output_node
        # Weight for each
        self.weight_hidden = NN_INPUT + 1
        self.weight_output = hidden_node + 1
        # Momentum and learning rate variable
        self.momentum = momentum
        self.rate = rate
        # Hidden node output
        self.output_layer_input = np.zeros(hidden_node + 1)
        # weights to every hidden and output node
        self.hidden_layer = ((np.random.rand(hidden_node, self.weight_hidden) / 10) - 0.05)
        self.output_layer = ((np.random.rand(output_node, self.weight_output) / 10) - 0.05)
        # Outputs layer
        self.hidden_layer_output = np.zeros(hidden_node)
        self.output_layer_output = np.zeros(output_node)
        # Errors layer
        self.hidden_layer_errors = np.zeros(hidden_node)
        self.output_layer_errors = np.zeros(output_node)
        # The last set of weight change
        self.hidden_layer_last_delta = np.zeros((self.hidden_node, self.weight_hidden))
        self.output_layer_last_delta = np.zeros((self.output_node, self.weight_output))
     
    # reset hidden layer output layer and confusion matrix
    def reset(self):
        self.hidden_layer_last_delta = np.zeros(self.hidden_node, self.weight_hidden)
        self.output_layer_last_delta = np.zeros(self.output_node, self.weight_output)
        self.confusion_matrix = np.zeros((10, 10), dtype=int)

    # Run each epoch
    def run_epoch(self, data, train, matrix):
        correct = 0
        total = len(data)
        for x in data:
            if self.run_trial(x, train, matrix):
                correct += 1
        return correct / total
    # result generator
    def generate_result(self, data):
        self.hidden_layer_output = sig_func((self.hidden_layer @ data))
        self.output_layer_input = np.insert(self.hidden_layer_output, 0, [1], axis=0)
        self.output_layer_output = sig_func(self.output_layer @ self.output_layer_input)
    # train the network
    def train(self, target, data):

        self.output_layer_errors = (self.output_layer_output * (1 - self.output_layer_output) *
                                    (self.targets[int(target)] - self.output_layer_output))

        sum_of_weights_to_out = self.output_layer.transpose() @ self.output_layer_errors
        self.hidden_layer_errors = (self.hidden_layer_output * (1 - self.hidden_layer_output) *
                                    sum_of_weights_to_out[1:])

        self.output_layer_last_delta = (((self.output_layer_errors[np.newaxis, :].transpose() @
                                          self.output_layer_input[np.newaxis, :]) * self.rate) +
                                        (self.output_layer_last_delta * self.momentum))

        self.hidden_layer_last_delta = (((self.hidden_layer_errors[np.newaxis, :].transpose() @ data[np.newaxis, :]) *
                                         self.rate) + (self.hidden_layer_last_delta * self.momentum))

        self.hidden_layer += self.hidden_layer_last_delta
        self.output_layer += self.output_layer_last_delta
    # run each value of each experiment
    def run_trial(self, trial_data, train, matrix):
        target = trial_data[0]

        self.generate_result(trial_data[1:])
        result = np.argmax(self.output_layer_output)

        if train:
            self.train(target, trial_data[1:])

        if matrix:
            self.confusion_matrix[int(result), int(target)] += 1

        return target == result
