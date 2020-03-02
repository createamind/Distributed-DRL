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

agent_weights = "agents/103.573526M_6.7_weights.pickle"#102.624116M_6.96_weights.pickle"#103.085962M_6.98_weights.pickle"#102.100236M_6.76_weights.pickle"#102.760798M_7.3_weights.pickle"#94.54844M_6.18_weights.pickle"#98.614691M_6.9_weights.pickle"#99.487003M_7.32_weights.pickle"#98.49637M_6.78_weights.pickle"#97.238274M_6.6_weights.pickle"#95.367972M_7.7_weights.pickle"#96.853064M_7.12_weights.pickle"#94.470649M_7.52_weights.pickle"#95.509652M_7.22_weights.pickle"#93.552998M_7.2_weights.pickle"#93.171218M_4.48_weights.pickle"#92.399664M_7.84_weights.pickle"#89.954949M_7.14_weights.pickle"#93.949107M_5.56_weights.pickle"#93.750531M_5.1_weights.pickle"#93.549288M_5.82_weights.pickle"#93.349858M_4.94_weights.pickle"#93.153151M_4.2_weights.pickle"#88.816672M_4.46_weights.pickle"#89.594806M_4.34_weights.pickle"#90.580364M_5.74_weights.pickle"#91.578249M_4.74_weights.pickle"#92.355662M_6.02_weights.pickle"#93.549288M_5.82_weights.pickle"#94.54844M_6.18_weights.pickle"#95.152022M_4.92_weights.pickle"#96.319342M_4.48_weights.pickle"#97.279252M_5.0_weights.pickle"#98.242557M_4.0_weights.pickle"#99.783999M_4.62_weights.pickle"#104.227424M_4.76_weights.pickle"#106.361906M_4.92_weights.pickle"#106.755215M_5.18Max_weights.pickle"#106.948044M_4.06_weights.pickle"#103.614035M_2.8_weights.pickle"#103.634041M_4.8Max_weights.pickle"#102.912908M_4.24_weights.pickle"#103.061805M_4.02_weights.pickle"#102.55931M_3.54Max_weights.pickle"#102.483514M_3.74Max_weights.pickle"
#"rightrandom0.015_buffersize2.5e6_ar3_rs150_COPY10M_2/87.662722M_6.38_weights.pickle"
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
    
    retl=retr=0

    test_env = football_env.create_environment(env_name=opt.env_name, stacked=opt.stacked,representation=opt.representation, render=False)

    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0

    cnt_pause = 0
    while not d:
        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        # handling env stuck
        if cnt_pause > 0 and o[108]:
            print("env getting stuck.....", cnt_pause, 'steps')
            cnt_pause = 0
        if not o[108]:
            cnt_pause += 1
            if cnt_pause <= 50:
                action = agent.get_action(o, False)
            else:
                action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.03)
        ep_ret += r
        ep_len += 1
        
        if r>0:
            retl+=1
        if r<0:
            retr+=1

    if j%10 == 0:
        print("agent_weights:", agent_weights)

    print("score: ",ep_ret,"[",retl,":",retr, "]")

    #print("test reward:", ep_ret, ep_len)
    rew.append(ep_ret)
    print("ave test reward:", sum(rew) / j, j)

print("agent_weights:", agent_weights)
print("ave test_reward:", sum(rew) / n)
