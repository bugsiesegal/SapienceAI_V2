from random import random, choice

import neat
from neat.attributes import StringAttribute, FloatAttribute, BoolAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.six_util import iteritems

from attributes import IntAttribute


class NeuronGene(BaseGene):
    __gene_attributes__ = [
        StringAttribute('neuron_type'),
        IntAttribute('sensory_index'),
        IntAttribute('action_index')
    ]

    def distance(self, other, config):
        d = 0
        if self.neuron_type != other.neuron_type:
            d += 1
        if self.sensory_index != other.sensory_index:
            d += 1
        if self.action_index != other.action_index:
            d += 1

        return d


class AxonGene(BaseGene):
    __gene_attributes__ = [
        FloatAttribute('weight'),
        FloatAttribute('activation_potential'),
        BoolAttribute('enabled')
    ]

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        d += abs(self.activation_potential - other.activation_potential)

        return d * config.compatibility_weight_coefficient


class BrainGenomeConfig:
    __params = [
        ConfigParameter('num_inputs', int),
        ConfigParameter('num_outputs', int),
        ConfigParameter('compatibility_disjoint_coefficient', float),
        ConfigParameter('compatibility_weight_coefficient', float),
        ConfigParameter('axon_add_prob', float),
        ConfigParameter('axon_delete_prob', float),
        ConfigParameter('neuron_add_prob', float),
        ConfigParameter('neuron_delete_prob', float)
    ]

    def __init__(self, params):
        self.__params += NeuronGene.get_config_params()
        self.__params += AxonGene.get_config_params()

        for p in self.__params:
            setattr(self, p.name, p.interpret(params))

            # By convention, input pins have negative keys, and the output
            # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

    def save(self, f):
        write_pretty_params(f, self, self.__params)

class BrainGenome:
    @classmethod
    def parse_config(cls, param_dict):
        return BrainGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        self.axons = {}
        self.neurons = {}

        self.fitness = None

    def mutate(self, config):
        if random() < config.neuron_add_prob:
            self.mutate_add_neuron(config)

        if random() < config.neuron_delete_prob:
            self.mutate_delete_neuron(config)

        if random() < config.axon_add_prob:
            self.mutate_add_axon(config)

        if random() < config.axon_delete_prob:
            self.mutate_delete_axon(config)

        for ag in self.axons.values():
            ag.mutate(config)

        for ng in self.neurons.values():
            ng.mutate(config)

    def configure_crossover(self, genome1, genome2, config):
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        for key, ag1 in iteritems(parent1.axons):
            ag2 = parent2.axons.get(key)
            if ag2 is None:
                self.axons[key] = ag1.copy()
            else:
                self.axons[key] = ag1.crossover(ag2)

        parent1_set = parent1.neurons
        parent2_set = parent2.neurons

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.neurons
            if ng2 is None:
                self.neurons[key] = ng1.copy()
            else:
                self.neurons[key] = ng1.crossover(ng2)

    def get_new_neuron_key(self):
        new_id = 0
        while new_id in self.neurons:
            new_id += 1
        return new_id

    def mutate_add_neuron(self, config):
        if not self.axons:
            return None, None

        axon_to_split = choice(list(self.axons.values()))
        new_neuron_id = self.get_new_neuron_key()
        ng = self.create_neuron(config, new_neuron_id)
        self.neurons[new_neuron_id] = ng

        axon_to_split.enabled = False

        i, o = axon_to_split.key
        self.add_axon(config, i, new_neuron_id)
        self.add_axon(config, new_neuron_id, o)

    def add_axon(self, config, input_key, output_key):
        key = (input_key, output_key)
        axon = AxonGene(key)
        axon.init_attributes(config)
        self.axons[key] = axon


