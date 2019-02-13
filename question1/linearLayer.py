import numpy as np
from numpy import random


class Linear(object):

    def __init__(self, input_size, output_size, parameter_init="Zero"):
        self.input_size = input_size
        self.output_size = output_size
        self.parameter_init = parameter_init
        self.weights = np.zeros((output_size, input_size), dtype=np.float32)
        self.bias = np.zeros((output_size, 1), dtype=np.float32)

        self.initialize_weights(self.parameter_init)

    def initialize_weights(self, para_init):
        if para_init == "Zero":
            pass
        elif para_init == "Normal":
            self.weights = random.normal(
                loc=0.0, scale=1.0, size=(self.output_size, self.input_size))
        elif para_init == "Glorot":
            d = np.sqrt(6 / (self.input_size + self.output_size))
            self.weights = random.uniform(
                low=-d, high=d, size=(self.output_size, self.input_size))

    def forward(self, input):
        output = np.matmul(input, self.weights.T) + self.bias.T
        return output

    def backward(self):
        pass
