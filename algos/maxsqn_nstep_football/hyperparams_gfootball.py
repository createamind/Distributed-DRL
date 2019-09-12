import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env


class HyperParameters:
    def __init__(self):
        # parameters set

        self.env_name = "academy_3_vs_1_with_keeper"  #'academy_empty_goal' #
        self.rollout_env_name = self.env_name
        self.exp_name = '3v1_randomFalse_0.1_scale100_b'

        self.env_random = False
        self.deterministic = False

        if self.env_random:
            self.rollout_env_name = self.env_name + "_random"
        if self.deterministic:
            self.rollout_env_name = self.env_name + "_d_True"

        self.with_checkpoints = False

        # gpu memory fraction
        self.gpu_fraction = 0.2

        self.ac_kwargs = dict(hidden_sizes=[600, 400, 200])

        env_football = football_env.create_environment(env_name=self.env_name,
                                                       with_checkpoints=False, representation='simple115',
                                                       render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_dim = env.observation_space.shape[0]
        self.obs_dim = 44 # 32# 51
        # self.obs_space = env.observation_space
        self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = 200000

        self.num_learners = 1
        self.num_workers = 12
        self.a_l_ratio = 3

        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.25

        self.gamma = 0.997
        self.replay_size = int(2e6)

        self.lr = 5e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 300
        self.start_steps = int(30000/self.num_workers)
        self.max_ep_len = 110
        self.save_freq = 1

        self.seed = 0

        self.summary_dir = './tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = './' + self.exp_name    # Directory for storing trained model
        self.is_restore = False


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        r = 0.0
        for _ in range(3):
            obs, reward, done, info = self._env.step(action)

            if obs[0] < 0.0:
                done = True
            if reward < 0:
                reward = 0
            r += reward

            if done:
                return obs, r * 100, done, info

        return obs, r * 100, done, info
