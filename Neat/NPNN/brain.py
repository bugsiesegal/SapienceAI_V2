import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import pandas as pd

class Brain:
    def __init__(self):
        self.axons = []
        self.neurons = []

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def add_axon(self, axon):
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

    def plot(self) -> None:
        df = pd.DataFrame(columns=["Input", "Output", "InpNeuron", "Weight"])

        fig, ax = plt.subplots(figsize=(15, 8))

        for axon in self.axons:
            df.loc[len(df.index)] = [int(id(axon.input_neuron)), int(id(axon.output_neuron)), axon.input_neuron.neuron_type, axon.output_neuron.neuron_type]

        print(df)

        G = nx.from_pandas_edgelist(df, source="Input", target="Output", create_using=nx.Graph())
        nx.draw(G)
        fig.show()

