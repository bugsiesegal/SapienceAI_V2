import multiprocessing
import time
from functools import partial
from multiprocessing import Pool

import gym
import numpy as np

from Neat.NPNN.brain import create_brain, Brain


def compute_fitness(genome, config, episodes, min_reward, max_reward):
    genome = genome[1]
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]
    reward_error = []
    brain = create_brain(genome, config).to_network()
    for score, data in episodes:
        dr = np.convolve(data[:, -1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:27]

            for i in range(10):
                output = brain.step(observation)

            reward_error.append(float((sum(output) - dr) ** 2))

    return reward_error


class ParallelEvaluator:
    def __init__(self, num_workers, timeout=None):
        self.num_workers = num_workers
        self.eval_function = self.fitness_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

        self.test_episodes = []

        self.episode_score = []
        self.episode_length = []

    # def __del__(self):
    #     self.pool.close()
    #     self.pool.join()

    def fitness_function(self, genome, config):
        scores = []
        env = gym.make("Ant-v3")
        brain = create_brain(genome, config).to_network()
        step = 0
        observation = env.reset()
        data = []
        for _ in range(1000):
            step += 1
            for __ in range(10):
                action = brain.step(observation[:27])
            observation, reward, done, info = env.step(action)
            data.append(np.hstack((observation, action, reward)))

            if done:
                observation = env.reset()
                break

        data = np.array(data)
        score = np.sum(data[:, -1])
        scores.append(score)

        test_episode = (score, data)

        return score, step, test_episode

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def eval_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        genome_list = []
        for genome_id, genome in genomes:
            genome_list.append(genome)

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        if self.generation % 10 == 1:
            funct = partial(self.eval_function, config=config)

            out = self.pool.map(funct, genome_list)

            for score, step, test_episode in out:
                self.episode_score.append(score)
                self.episode_length.append(step)
                self.test_episodes.append(test_episode)

        print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.num_workers < 2:
            for genome in genomes:
                reward_error = compute_fitness(genome, config, self.test_episodes, self.min_reward, self.max_reward)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome in genomes:
                    jobs.append(pool.apply_async(compute_fitness,
                                                 (genome, config, self.test_episodes,
                                                  self.min_reward, self.max_reward)))

                for job, (genome_id, genome) in zip(jobs, genomes):
                    reward_error = job.get(timeout=None)
                    genome.fitness = -np.sum(reward_error) / len(self.test_episodes)

        print("final fitness compute time {0}\n".format(time.time() - t0))
