

from hyperparams_gfootball import HyperParameters
from actor_learner import Actor, Learner

import pickle
import gfootball.env as football_env
import time

class FootballWrapper(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


    def step(self, action):
        r = 0.0
        for _ in range(2):
            obs, reward, done, info = self._env.step(action)
            r += reward

            if done:
                return obs, r, done, info

        return obs, r, done, info


opt = HyperParameters()
# opt.hidden_size = (400, 300)

agent = Actor(opt, job="test") # 11v11_0.1_Ln5_rp2_mp1/1M_weights.pickle # random_easy_new_a/31M_weights.pickle
keys, weights = agent.get_weights()  # 11v11_easy_Ln5_rp2_-1done_scale225_exp2/ # 11v11_easy_new642_33_200
pickle_in = open("./data/debug_11v11_easy_343_200_done_vqloss_clip10/1M_weights.pickle", "rb") # lazy_0.1/7M_weights.pickle  11v11_incentive_0.1/Maxret_weights.pickle
weights = pickle.load(pickle_in)


weights = [weights[key] for key in keys]

agent.set_weights(keys, weights)

test_env = football_env.create_environment(env_name="11_vs_11_easy_stochastic",   # academy_single_goal_versus_lazy  11_vs_11_stochastic_random
                                           representation='simple115', render=True)

# test_env = FootballWrapper(test_env)

num = 100

ave_ep_ret = 0.0
for j in range(num):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    while not d: #( d or (ep_len == opt.max_ep_len)):

        action = agent.get_action(o, True)
        # action = test_env.action_space.sample()

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.01)
        ep_ret += r
        ep_len += 1

    ave_ep_ret = (j * ave_ep_ret + ep_ret) / (j + 1)
    print('ep_len', ep_len, 'ep_ret:', ep_ret, 'ave_ep_ret:', ave_ep_ret, '({}/{})'.format(j + 1, num))
