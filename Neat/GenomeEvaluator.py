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
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
