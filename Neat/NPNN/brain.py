import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas
from matplotlib.ticker import AutoMinorLocator
from neat.six_util import iteritems
from pyvis.network import Network
import pandas as pd

from NPNN.axon import Axon
from NPNN.neuron import Neuron

import matplotlib as mpl

import wandb


class Brain:
    def __init__(self):
        self.axons = []
        self.neurons = []

    def add_neuron(self, neuron):
        neuron.h_id = len(self.neurons)
        self.neurons.append(neuron)

    def add_axon(self, axon):
        axon.h_id = len(self.axons)
        self.axons.append(axon)

    def step(self, input_array, propagations_per_step=6):
        state = []
        for i in range(propagations_per_step):
            # print(input_array)
            output_array = []

            for axon in self.axons:
                axon.check_activation()

            for neuron in self.neurons:
                output_array.append(neuron.step(input_array))

            output_array = list(np.asarray(output_array).sum(axis=0))

            for axon in self.axons:
                axon.propagate()

        return output_array

    def make_graph(self):
        G = nx.DiGraph()
        neuron_to_id = {}

        for node in range(len(self.neurons)):
            neuron_to_id[self.neurons[node]] = node
            G.add_node(node, energy=self.neurons[node].energy, neuron_type=self.neurons[node].neuron_type)

        for edge in self.axons:
            G.add_edge(neuron_to_id[edge.input_neuron], neuron_to_id[edge.output_neuron],
                       activation_potential=edge.activation_potential,
                       axon_weight=edge.weight)

        self.pos = nx.spring_layout(G)

        self.G = G

    def plot(self, i):
        node_energies = [i.energy for i in self.neurons]
        cmap = plt.cm.plasma

        nodes = nx.draw_networkx_nodes(self.G, pos=self.pos, node_color=node_energies, cmap=cmap)
        edges = nx.draw_networkx_edges(self.G, pos=self.pos, arrowstyle="->")

        nx.draw_networkx_labels(self.G, pos=self.pos, labels=nx.get_node_attributes(self.G, "neuron_type"))

        plt.colorbar(nodes)

        ax = plt.gca()
        ax.set_axis_off()
        plt.savefig("figures/%s.svg" % i)
        plt.show()


def create_brain(genome, config) -> Brain:
    brain = Brain()
    neurons = {}

    for key, n in iteritems(genome.neurons):
        neuron = Neuron(config.genome_config.num_outputs, n.neuron_type, n.action_index, n.sensory_index)
        brain.add_neuron(neuron)
        neurons[key] = neuron

    for key, a in iteritems(genome.axons):
        if a.enabled:
            brain.add_axon(Axon(neurons[a.key[0]], neurons[a.key[1]], a.activation_potential, a.weight))

    return brain
