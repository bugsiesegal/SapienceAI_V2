import os

import neat
from neat.six_util import iteritems

from NPNN.axon import Axon
from NPNN.brain import Brain
from NPNN.neuron import Neuron
from Neat.Genome import BrainGenome


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


def eval_genomes(genomes, config):
    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]
    for genome_id, genome in genomes:
        brain = create_brain(genome, config)
        genome.fitness = 4.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            print(xi)
            # Number of Propagations
            for i in range(7):
                brain.step(xi)

            if brain.step(xi)[0] == xo[0]:
                print(brain.step(xi))

            genome.fitness -= (brain.step(xi)[0] - xo[0]) ** 2


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
    winner = p.run(eval_genomes, 20)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create_brain(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = None
        for i in range(7):
            output = winner_net.step(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
