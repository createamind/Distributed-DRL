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



opt = HyperParameters()

agent = Actor(opt, job="test")
keys, weights = agent.get_weights()
pickle_in = open("./data/11v11_incentive_0.1/Maxret_weights.pickle", "rb")
weights = pickle.load(pickle_in)


weights = [weights[key] for key in keys]

agent.set_weights(keys, weights)

test_env = football_env.create_environment(env_name="11_vs_11_stochastic_random",
                                           representation='simple115', render=True)

num = 100

ave_ep_ret = 0.0
for j in range(num):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not ( d or (ep_len == opt.max_ep_len)):

        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        time.sleep(0.03)
        ep_ret += r
        ep_len += 1

    ave_ep_ret = (j * ave_ep_ret + ep_ret) / (j + 1)
    print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:', ave_ep_ret, '({}/{})'.format(j + 1, num))
