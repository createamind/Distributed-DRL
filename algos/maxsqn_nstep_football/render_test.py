import numpy as np
import tensorflow as tf
import time
import ray
import gym

from hyperparams_gfootball import HyperParameters
from actor_learner import Actor, Learner

import os
import pickle
import gfootball.env as football_env

from ray.rllib.utils.compression import pack, unpack

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# "Pendulum-v0" 'BipedalWalker-v2' 'LunarLanderContinuous-v2'
flags.DEFINE_string("env_name", "11_vs_11_stochastic_random_1", "game env")
flags.DEFINE_string("exp_name", "Exp1", "experiments name")
flags.DEFINE_integer("total_epochs", 500, "total_epochs")
flags.DEFINE_integer("num_workers", 6, "number of workers")
flags.DEFINE_integer("num_learners", 1, "number of learners")
flags.DEFINE_string("weights_file", "", "empty means False. "
                                        "[/path/Maxret_weights.pickle] means restore weights from this pickle file.")
flags.DEFINE_float("a_l_ratio", 2, "steps / sample_times")

opt = HyperParameters(FLAGS.env_name, FLAGS.exp_name, FLAGS.num_workers, FLAGS.a_l_ratio,
                      FLAGS.weights_file)
opt.hidden_size = (300, 400, 300)
# opt.hidden_size = (400, 300)

agent = Actor(opt, job="test")
keys, weights = agent.get_weights()
pickle_in = open("RMax_weights.pickle", "rb")
weights_all = pickle.load(pickle_in)

weights = [weights_all[key] for key in keys]

agent.set_weights(keys, weights)

test_env = football_env.create_environment(env_name=opt.env_name, stacked=opt.stacked,
                                           representation=opt.representation, render=True)

# test_env = FootballWrapper(test_env)
n = 100

rew = []
for j in range(1, n + 1):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not d:
        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.03)
        ep_ret += r
        ep_len += 1

    # print("test reward:", ep_ret, ep_len)
    # exit()
    rew.append(ep_ret)
    print("ave test reward:", sum(rew) / j, j)
print("ave test_reward:", sum(rew) / n)


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(3):
            obs, reward, done, info = self._env.step(action)

            r += reward

            if done:
                return obs, r, done, info

        return obs, r, done, info
