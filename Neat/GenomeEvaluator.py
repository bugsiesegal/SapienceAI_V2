from functools import partial
from multiprocessing import Pool


class ParallelEvaluator:
    def __init__(self, num_workers, eval_function, timeout=None):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def eval_genomes(self, genomes, config):
        genome_list = []
        for genome_id, genome in genomes:
            genome_list.append(genome)

        funct = partial(self.eval_function, config=config)

        results = self.pool.map(funct, genome_list)

        for result, (genome_id, genome) in zip(results, genomes):
            genome.fitness = result
