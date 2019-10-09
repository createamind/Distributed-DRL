import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env
from numbers import Number


class HyperParameters:
    def __init__(self, env_name, exp_name, total_epochs, num_workers, a_l_ratio, weights_file):
        # parameters set

        self.env_name = "11_vs_11_easy_stochastic"

        # "_random", "_d_True", ""
        self.rollout_env_name = self.env_name + ""
        self.exp_name = str(exp_name)

        self.with_checkpoints = False

        self.representation = 'simple115'
        self.stacked = False
        self.model = "mlp"

        self.a_l_ratio = a_l_ratio
        self.weights_file = weights_file

        # gpu memory fraction
        self.gpu_fraction = 0.2

        # self.ac_kwargs = dict(hidden_sizes=[400, 600, 400, 200])
        self.ac_kwargs = dict(hidden_sizes=[600, 400, 200])

        env_football = football_env.create_environment(env_name=self.env_name, stacked=self.stacked,
                                                       representation=self.representation, render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_dim = env.observation_space.shape
        #
        # # self.obs_dim = 51
        # self.obs_space = env.observation_space
        # # self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # # TODO gfootball 1.3
        # if len(self.obs_dim) == 1 and self.env_name == "academy_3_vs_1_with_keeper":
        #     # self.obs_dim = (self.obs_dim,)
        #     self.obs_dim = (51, )
        #     self.obs_space = Box(low=-1.0, high=1.0, shape=(51,), dtype=np.float32)
        #
        # self.act_dim = env.action_space.n
        # self.act_space = env.action_space

        scenario_obsdim = {'11_vs_11_easy_stochastic': 115, 'academy_empty_goal': 32, 'academy_empty_goal_random': 32,
                           'academy_3_vs_1_with_keeper': 51,
                           'academy_3_vs_1_with_keeper_random': 51, 'academy_single_goal_versus_lazy': 108}
        self.obs_dim = scenario_obsdim[self.env_name]
        self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.o_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        self.a_shape = self.act_space.shape

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.Ln = 3
        self.use_max = False
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.gamma = 0.997
        self.replay_size = int(3e6)

        self.lr = 5e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 256
        self.start_steps = int(3e4)
        self.max_ep_len = 960
        self.save_freq = 1

        self.max_ret = 0

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name    # Directory for storing trained model

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name


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
                return obs, r * 200, done, info

        return obs, r * 200, done, info
