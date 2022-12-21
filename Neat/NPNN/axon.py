from .neuron import Neuron


class Axon:
    input_neuron: "Neuron"
    output_neuron: "Neuron"

    """
    Simulates Axon.
    """
    def __init__(self, input_neuron: Neuron, output_neuron: Neuron, activation_potential: float, weight: float, time_delay: int):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron

        self.activation_potential = activation_potential
        self.weight = weight

        self.time_delay = time_delay

        self.buffer = []

        for i in range(time_delay):
            self.buffer.append(False)

        self.h_id = None

    def propagate(self):
        if self.buffer[0]:
            self.output_neuron.energy += self.weight
        self.buffer = self.buffer[1:]

    def check_activation(self):
        if self.input_neuron.energy >= self.activation_potential:
            self.buffer.append(True)
        else:
            self.buffer.append(False)
