import numpy as np
import tensorflow as tf
import time
import ray
import gym

from hyperparams import HyperParameters
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy
import signal


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# "Pendulum-v0" 'BipedalWalker-v2' 'LunarLanderContinuous-v2'
flags.DEFINE_string("env_name", "BipedalWalkerHardcore-v2", "game env")
flags.DEFINE_integer("total_epochs", 500, "total_epochs")
flags.DEFINE_integer("num_workers", 1, "number of workers")
flags.DEFINE_integer("num_learners", 1, "number of learners")
flags.DEFINE_string("is_restore", "False", "True or False. True means restore weights from pickle file.")
flags.DEFINE_float("a_l_ratio", 10, "steps / sample_times")

opt = HyperParameters(FLAGS.env_name, FLAGS.total_epochs, FLAGS.num_workers, FLAGS.a_l_ratio)

agent = Actor(opt, job="main")
keys, weights = agent.get_weights()
pickle_in = open("weights.pickle", "rb")
weights = pickle.load(pickle_in)


weights = [weights[key] for key in keys]

agent.set_weights(keys, weights)

test_env = gym.make(opt.env_name)

n = 2

rew = []
for j in range(n):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not (d or (ep_len == opt.max_ep_len)):
        # Take deterministic actions at test time
        test_env.render()
        action = agent.get_action(o, True)
        print(action)
        o, r, d, _ = test_env.step(action)
        ep_ret += r
        ep_len += 1
    rew.append(ep_ret)
print("test_reward:", sum(rew)/n)
