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

    def step(self, input_array):
        # print(input_array)
        output_array = []

        for neuron in self.neurons:
            output_array.append(neuron.step(input_array))

        output_array = list(np.asarray(output_array).sum(axis=0))

        for axon in self.axons:
            axon.propagate()

        return output_array

    def to_df(self) -> pandas.DataFrame:
        df = pd.DataFrame(columns=["Input", "Output", "InpNeuron", "OutNeuron", "Weight", "Activation Potential"])

        for axon in self.axons:
            df.loc[len(df.index)] = [axon.input_neuron, axon.output_neuron,
                                     axon.input_neuron.neuron_type, axon.output_neuron.neuron_type, axon.weight,
                                     axon.activation_potential]
        return df

    def plot(self):
        G = nx.DiGraph()

        nodes = []
        for neuron in self.neurons:
            if neuron.neuron_type == 2:
                nodes.append((neuron.h_id, {"color": "green", "label": "Output: " + str(neuron.h_id)}))
            elif neuron.neuron_type == 1:
                nodes.append((neuron.h_id, {"color": "red", "label": "Input: " + str(neuron.h_id)}))
            else:
                nodes.append((neuron.h_id, {"color": "blue", "label": str(neuron.h_id)}))
        G.add_nodes_from(nodes)

        edges = []
        for axon in self.axons:
            edges.append((axon.input_neuron.h_id, axon.output_neuron.h_id, {
                'axon_label': 'Weight: {:.2f}, Potential: {:.2f}'.format(axon.weight, axon.activation_potential)}))
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, scale=0.25)

        nx.draw(G, pos)
        node_labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'axon_label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=4)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=4)
        plt.savefig("C:\\Users\\Jake\\PycharmProjects\\SapienceAI_V2\\Neat\\Model_Images\\brain-structure.png", dpi=400)
        plt.show()

        wandb.log(
            {"brain structure": wandb.Image(
                "C:\\Users\\Jake\\PycharmProjects\\SapienceAI_V2\\Neat\\Model_Images\\brain-structure.png")}
        )


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
