import neat
from neat.attributes import StringAttribute, FloatAttribute
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import BaseGene

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
        FloatAttribute('activation_potential')
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



