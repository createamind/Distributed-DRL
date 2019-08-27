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


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# "Pendulum-v0" 'BipedalWalker-v2' 'LunarLanderContinuous-v2'
flags.DEFINE_string("env_name", "BipedalWalkerHardcore-v2", "game env")
flags.DEFINE_integer("total_epochs", 500, "total_epochs")
flags.DEFINE_integer("num_workers", 1, "number of workers")
flags.DEFINE_integer("num_learners", 1, "number of learners")
flags.DEFINE_string("is_restore", "False", "True or False. True means restore weights from pickle file.")
flags.DEFINE_float("a_l_ratio", 10, "steps / sample_times")

opt = HyperParameters(FLAGS.total_epochs, FLAGS.num_workers, FLAGS.a_l_ratio)

agent = Actor(opt, job="test")
keys, weights = agent.get_weights()
pickle_in = open("68_weights.pickle", "rb")
weights = pickle.load(pickle_in)


weights = [weights[key] for key in keys]

agent.set_weights(keys, weights)

test_env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper", with_checkpoints=False,
                                           representation='simple115', render=True)

n = 100

rew = []
for j in range(n):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not (d or (ep_len == opt.max_ep_len)):

        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        ep_ret += r
        ep_len += 1
        time.sleep(0.03)
    print("test reward:", ep_ret, ep_len)
    rew.append(ep_ret)
print("ave test_reward:", sum(rew)/n)
