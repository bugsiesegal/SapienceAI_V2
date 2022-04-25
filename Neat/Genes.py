from neat.attributes import StringAttribute, FloatAttribute, BoolAttribute
from neat.genes import BaseGene


class NeuronGene(BaseGene):
    __gene_attributes__ = [
        StringAttribute('neuron_type'),
        StringAttribute('sensory_index'),
        StringAttribute('action_index')
    ]

    def distance(self, other, config):
        d = 0

        if self.neuron_type == '0' and other.neuron_type == '0':
            d = 0
        elif self.neuron_type == '1' and other.neuron_type == '1':
            if self.sensory_index == other.sensory_index:
                d = 0
            else:
                d = 1
        else:
            if self.action_index == other.action_index:
                d = 0
            else:
                d = 1

        return d * config.compatibility_weight_coefficient


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