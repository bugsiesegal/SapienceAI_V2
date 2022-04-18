import os
from itertools import chain, product
from random import random, choice

import neat
from neat.attributes import StringAttribute, FloatAttribute, BoolAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene
from neat.six_util import iteritems, iterkeys

from attributes import IntAttribute

from NPNN.brain import Brain
from NPNN.axon import Axon
from NPNN.neuron import Neuron


class NeuronGene(BaseGene):
    __gene_attributes__ = [
        StringAttribute('neuron_type'),
        StringAttribute('sensory_index'),
        StringAttribute('action_index')
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
        self.input_keys = [i for i in range(self.num_inputs)]
        self.output_keys = [self.num_inputs+i for i in range(self.num_outputs)]

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
            elif ng2.key != ng1.key:
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


    def mutate_add_axon(self, config):
        possible_outputs = list(iterkeys(self.neurons))
        out_neuron = choice(possible_outputs)

        possible_inputs = possible_outputs
        in_neuron = choice(possible_inputs)

        if in_neuron == out_neuron:
            return

        self.add_axon(config, in_neuron, out_neuron)

    def mutate_delete_neuron(self, config):
        available_neurons = [(k, v) for k, v in iteritems(self.neurons) if k not in config.output_keys]
        if not available_neurons:
            return -1

        del_key, del_neuron = choice(available_neurons)

        axons_to_delete = set()
        for k, v in iteritems(self.axons):
            if del_key in v.key:
                axons_to_delete.add(v.key)

        for key in axons_to_delete:
            del self.axons[key]

        del self.neurons[del_key]

        return del_key

    def mutate_delete_axon(self, config):
        if self.axons:
            key = choice(list(self.axons.keys()))
            del self.axons[key]

    def distance(self, other, config):
        neuron_distance = 0.0
        if self.neurons or other.neurons:
            disjoint_neurons = 0
            for k2 in iterkeys(other.neurons):
                if k2 not in self.neurons:
                    disjoint_neurons += 1

            for k1, n1 in iteritems(self.neurons):
                n2 = other.neurons.get(k1)
                if n2 is None:
                    disjoint_neurons += 1
                else:
                    neuron_distance += n1.distance(n2, config)

            max_neurons = max(len(self.neurons), len(other.neurons))
            neuron_distance = (
                                      neuron_distance + config.compatibility_disjoint_coefficient * disjoint_neurons) / max_neurons

        axon_distance = 0.0
        if self.axons or other.axons:
            disjoint_axons = 0
            for k2 in iterkeys(other.axons):
                if k2 not in self.axons:
                    disjoint_axons += 1

            for k1, c1 in iteritems(self.axons):
                c2 = other.axons.get(k1)
                if c2 is None:
                    disjoint_axons += 1
                else:
                    # Homologous genes compute their own distance value.
                    axon_distance += c1.distance(c2, config)

            max_axon = max(len(self.axons), len(other.axons))
            axon_distance = (
                                    axon_distance + config.compatibility_disjoint_coefficient * disjoint_axons) / max_axon

        distance = neuron_distance + axon_distance

        return distance

    def size(self):
        """Returns genome 'complexity', taken to be (number of neurons, number of enabled axons)"""
        num_enabled_axons = sum([1 for cg in self.axons.values() if cg.enabled is True])
        return len(self.neurons), num_enabled_axons

    def __str__(self):
        s = "neurons:"
        for k, ng in iteritems(self.neurons):
            s += "\n\t{0} {1!s}".format(k, ng)
        s += "\naxons:"
        axons = list(self.axons.values())
        axons.sort()
        for c in axons:
            s += "\n\t" + str(c)
        return s

    def add_neuron(self, config):
        for i in range(config.num_hidden):
            neuron_key = self.get_new_neuron_key()
            assert neuron_key not in self.neurons
            neuron = self.__class__.create_neuron(config, neuron_key)
            self.neurons[neuron_key] = neuron

    def configure_new(self, config):
        for input_neuron_key in config.input_keys:
            self.neurons[input_neuron_key] = NeuronGene(key=0)
            self.neurons[input_neuron_key].init_attributes(config)
            self.neurons[input_neuron_key].__setattr__('action_index', '0')
            self.neurons[input_neuron_key].__setattr__('sensory_index', str(input_neuron_key))
            self.neurons[input_neuron_key].__setattr__('neuron_type', '1')

        for output_neuron_key in config.output_keys:
            self.neurons[output_neuron_key] = NeuronGene(key=0)
            self.neurons[output_neuron_key].init_attributes(config)
            self.neurons[output_neuron_key].__setattr__('action_index', '0')
            self.neurons[output_neuron_key].__setattr__('sensory_index', str(input_neuron_key))
            self.neurons[output_neuron_key].__setattr__('neuron_type', '2')


        for ia, oa in product(config.input_keys, config.output_keys):
            self.axons[(ia, oa)] = AxonGene((ia, oa))
            self.axons[(ia, oa)].init_attributes(config)


    @staticmethod
    def create_neuron(config, neuron_id):
        neuron = NeuronGene(neuron_id)
        neuron.init_attributes(config)
        return neuron

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = AxonGene((input_id, output_id))
        connection.init_attributes(config)
        return connection


def create_brain(genome, config) -> Brain:
    brain = Brain()
    neurons = {}

    for key, n in iteritems(genome.neurons):
        neuron = Neuron(config.genome_config.num_outputs, n.neuron_type, n.action_index, n.sensory_index)
        brain.add_neuron(neuron)
        neurons[key] = neuron

    for key, a in iteritems(genome.axons):
            brain.add_axon(Axon(neurons[a.key[0]], neurons[a.key[1]], a.activation_potential, a.weight))

    return brain

def eval_genomes(genomes, config):
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
    for genome_id, genome in genomes:
        brain = create_brain(genome, config)
        genome.fitness = 0
        for xi, xo in zip(xor_inputs, xor_outputs):
            out = []
            for i in range(20):
                out = brain.step(xi)

            genome.fitness -= (out[0] - xo[0])**2


def run(config_file):
    # Load configuration.
    config = neat.Config(BrainGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 200)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    xor_inputs = [(0.0,0.0),(0.0, 1.0), (1.0, 0.0), (1.0,1.0)]
    xor_outputs = [(0.0), (1.0,), (1.0,),(1.0)]

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create_brain(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = None
        for i in range(5):
            output = winner_net.step(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)

