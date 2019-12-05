import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gym
from numbers import Number


class HyperParameters:
    def __init__(self, env_name, exp_name, num_workers, a_l_ratio, weights_file):
        # parameters set

        self.exp_name = exp_name

        self.env_name = env_name
        # "_random", "_d_True", ""
        self.rollout_env_name = self.env_name + ""

        self.model = "mlp"
        assert self.model in ["mlp", "cnn"], "model must be mlp or cnn!"
        if self.model == "cnn":
            self.representation = "extracted"
            self.stacked = True
        else:
            self.representation = 'simple115'
            self.stacked = False

        self.a_l_ratio = a_l_ratio
        self.weights_file = weights_file
        self.start_steps = int(5e4)
        if self.weights_file:
            self.start_steps = int(10e6)

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.hidden_size = (300, 400, 300)

        self.obs_noise = 0
        self.act_noise = 0.3
        self.reward_scale = 5
        env = Wrapper(gym.make(self.env_name), self.obs_noise, self.act_noise, self.reward_scale, 3)

        # env = FootballWrapper(env_football)

        # self.obs_space = Box(low=-1.0, high=1.0, shape=self.obs_dim, dtype=np.float32)
        self.obs_dim = env.observation_space.shape
        self.obs_space = env.observation_space
        self.obs_shape = self.obs_space.shape

        self.act_dim = env.action_space.shape
        self.act_space = env.action_space
        self.act_shape = self.act_space.shape

        self.num_workers = num_workers
        self.num_learners = 1

        self.use_max = False
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.use_bn = False
        self.c_regularizer = 0.0

        self.gamma = 0.997

        # self.num_buffers = 1
        self.num_buffers = self.num_workers // 25 + 1
        if self.model == 'cnn':
            self.buffer_size = int(3e4)
        else:
            self.buffer_size = int(3e6)

        self.buffer_size = self.buffer_size // self.num_buffers

        self.lr = 5e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 256

        self.Ln = 8
        self.action_repeat = 2

        self.max_ep_len = 2900
        self.save_freq = 1

        self.max_ret = 0

        self.epsilon = 0
        self.epsilon_alpha = 7

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name  # Directory for storing trained model
        self.save_interval = int(5e5)

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name


class Wrapper(object):

    def __init__(self, env, obs_noise, act_noise, reward_scale, action_repeat=3):
        self._env = env
        self.action_repeat = action_repeat
        self.act_noise = act_noise
        self.obs_noise = obs_noise
        self.reward_scale = reward_scale

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset() + self.obs_noise * (-2 * np.random.random(24) + 1)
        return obs

    def step(self, action):
        action += self.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_ = self._env.step(action)
            r = r + reward_
            # r -= 0.001
            if done_ and self.action_repeat != 1:
                return obs_ + self.obs_noise * (-2 * np.random.random(24) + 1), 0.0, done_, info_
            if self.action_repeat == 1:
                return obs_, r, done_, info_
        return obs_ + self.obs_noise * (-2 * np.random.random(24) + 1), self.reward_scale * r, done_, info_
