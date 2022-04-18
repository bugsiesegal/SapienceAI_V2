import numpy as np


class Brain:
    def __init__(self):
        self.axons = []
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_axon(self, axon):
        self.axons.append(axon)

    def step(self, input_array):
        output_array = []

        for neuron in self.neurons:
            output_array.append(neuron.step(*input_array))

        output_array = list(np.asarray(output_array).sum(axis=0))

        for axon in self.axons:
            axon.propagate()

        return output_array
