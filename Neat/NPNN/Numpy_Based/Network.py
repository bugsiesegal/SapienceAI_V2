from dataclasses import dataclass

from typing import Tuple
import numpy as np


class Brain:
    def __init__(self, num_neurons, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.total_neurons = num_neurons + input_size + output_size

        self.neuron_array = np.zeros(self.total_neurons)
        self.threshold_potential = np.zeros((self.total_neurons, self.total_neurons))
        self.action_potential = np.zeros((self.total_neurons, self.total_neurons))

    def step(self, input_array):
        input_array = np.array(input_array)
        tiled_array = np.tile(np.concatenate([input_array, self.neuron_array[self.input_size:]]),
                              (self.neuron_array.size, 1))
        activation_truth_table = tiled_array >= self.threshold_potential
        final_table = np.where(activation_truth_table, self.action_potential, np.zeros(self.action_potential.shape))
        self.neuron_array = np.sum(final_table, axis=1).reshape((-1,))
        # print(clamp(self.neuron_array[self.input_size:self.output_size + self.input_size]))
        return self.neuron_array[self.input_size:self.output_size + self.input_size]

    def randomize(self):
        self.threshold_potential = np.random.rand(*self.threshold_potential.shape) * 20 - 10
        self.action_potential = np.random.rand(*self.action_potential.shape) * 20 - 10

    def to_array(self):
        return np.stack([self.threshold_potential, self.action_potential])

    def from_array(self, arr):
        self.threshold_potential = np.split(arr, 2)[0]
        self.action_potential = np.split(arr, 2)[1]


def loss_function(x, y):
    return (x - y) ** 2
