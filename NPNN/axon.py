from typing import Dict

from NPNN.neuron import Neuron


class Axon:
    """
    Simulates Axon.
    """
    def __init__(self, input_neuron: Neuron, output_neuron: Neuron, activation_potential: float, weight: float):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron

        self.activation_potential = activation_potential
        self.weight = weight

    def propagate(self):
        if self.input_neuron.energy >= self.activation_potential:
            self.output_neuron.energy += self.weight