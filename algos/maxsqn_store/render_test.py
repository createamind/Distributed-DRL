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

# "1_vs_1_easy" '11_vs_11_easy_stochastic' '11_vs_11_competition'
flags.DEFINE_string("env_name", "11_vs_11_competition", "game env")
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

agent_weights = "rightrandom0.015_buffersize2.5e6_ar3_rs150_COPY10M_2/87.662722M_6.38_weights.pickle"
print("agent_weights:", agent_weights)

agent = Actor(opt, job="test")
keys, _ = agent.get_weights()
with open(agent_weights, "rb") as pickle_in:
    weights_all = pickle.load(pickle_in)
    weights = [weights_all[key] for key in keys]
    agent.set_weights(keys, weights)


# test_env = FootballWrapper(test_env)
n = 1000

rew = []
for j in range(1, n + 1):

    test_env = football_env.create_environment(env_name=opt.env_name, stacked=opt.stacked,representation=opt.representation, render=False)

    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not d:
        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.03)
        ep_ret += r
        ep_len += 1

    print("test reward:", ep_ret, ep_len)
    # exit()
    rew.append(ep_ret)
    print("ave test reward:", sum(rew) / j, j)
print("ave test_reward:", sum(rew) / n)
