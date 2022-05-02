import configparser
import os

import neat
import wandb
from neat.six_util import iteritems

from NPNN.axon import Axon
from NPNN.brain import Brain
from NPNN.neuron import Neuron
from Genome import BrainGenome

import time

from neat.math_util import mean, stdev
from neat.reporting import BaseReporter
from neat.six_util import itervalues, iterkeys


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
        genome.fitness = 0.0
        for xi, xo in zip(xor_inputs, xor_outputs):
            # Number of Propagations
            for i in range(8):
                brain.step(xi)

            genome.fitness -= ((abs(brain.step(xi)[0] - xo[0])) ** 2)


class WandbStdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)

        wandb.log({"number_species": ns})

        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag")
            print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

        wandb.log({"best_fitness": best_genome.fitness,
                   "average_fitness": fit_mean,
                   })

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)


def run(config_file):
    # Load configuration.
    config = neat.Config(BrainGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(WandbStdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(20))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 200)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = create_brain(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = None
        for i in range(20):
            output = winner_net.step(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


def param_tuning(config_path):
    configparse = configparser.ConfigParser()
    configparse.read(config_path)

    config_dict = {}
    for section in configparse.sections():
        for key in configparse[section]:
            config_dict[key] = configparse[section][key]

    wandb.init(project="SapienceAI_V2-Neat", entity="bugsiesegal", config=config_dict)

    config_dict = wandb.config

    for section in configparse.sections():
        for key in configparse[section]:
            if config_dict.get(key) is not None:
                configparse[section][key] = str(config_dict.get(key))

    with open(config_path, "w") as config_file:
        configparse.write(config_file)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')

    param_tuning(config_path)

    run(config_path)
