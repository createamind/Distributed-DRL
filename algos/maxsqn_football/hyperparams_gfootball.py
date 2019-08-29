import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env


class HyperParameters:
    def __init__(self, env_name, exp_name, total_epochs, num_workers=1, a_l_ratio=1):
        # parameters set

        self.env_name = "academy_3_vs_1_with_keeper"
        self.rollout_env_name = self.env_name
        self.exp_name = str(exp_name)

        self.env_random = False
        self.deterministic = True

        if self.env_random:
            self.rollout_env_name = self.env_name + "_random"
        if self.deterministic:
            self.rollout_env_name = self.env_name + "_d_True"

        self.with_checkpoints = False

        self.a_l_ratio = a_l_ratio

        # gpu memory fraction
        self.gpu_fraction = 0.2

        self.ac_kwargs = dict(hidden_sizes=[600, 400, 200])

        env_football = football_env.create_environment(env_name=self.env_name,
                                                       with_checkpoints=False, representation='simple115',
                                                       render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_dim = env.observation_space.shape[0]
        self.obs_dim = 51
        # self.obs_space = env.observation_space
        self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.alpha = 0.1

        self.gamma = 0.997
        self.replay_size = 3000000

        self.lr = 1e-4
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 256
        self.start_steps = 20000
        self.max_ep_len = 300
        self.save_freq = 1

        self.seed = 0

        self.summary_dir = './tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = './' + exp_name    # Directory for storing trained model


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        r = 0.0
        for _ in range(1):
            obs, reward, done, info = self._env.step(action)

            if obs[0] < 0.0:
                done = True
            if reward < 0:
                reward = 0
            r += reward

            if done:
                return obs, r * 200, done, info

        return obs, r * 200, done, info
