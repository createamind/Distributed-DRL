import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gym
from math import ceil


class HyperParameters:
    def __init__(self, env_name, exp_name, num_workers, a_l_ratio, weights_file):
        # parameters set

        self.exp_name = exp_name
        self.env_name = env_name

        self.model = "mlp"
        assert self.model in ["mlp", "cnn"], "model must be mlp or cnn!"

        self.a_l_ratio = a_l_ratio
        self.weights_file = weights_file

        self.recover = False
        self.checkpoint_freq = 21600  # 21600s = 6h

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.hidden_size = (400, 300)

        env_gym = gym.make(self.env_name)

        # env = FootballWrapper(env_football)
        env = env_gym

        # self.obs_space = Box(low=-1.0, high=1.0, shape=self.obs_dim, dtype=np.float32)
        self.obs_dim = env.observation_space.shape
        self.obs_space = env.observation_space
        self.obs_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space
        self.act_shape = self.act_space.shape

        self.num_workers = num_workers
        self.num_learners = 1

        self.use_max = False
        self.reward_scale = 100

        self.alpha = 0.2
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.use_bn = False
        self.c_regularizer = 0.0

        self.gamma = 0.99

        # self.num_buffers = 1
        if self.model == 'cnn':
            self.buffer_size = int(3e4)
        else:
            self.buffer_size = int(1e6)
        self.num_buffers = self.num_workers // 25 + 1
        self.buffer_size = self.buffer_size // self.num_buffers

        self.start_steps = int(1e3) // self.num_buffers
        if self.weights_file:
            self.start_steps = self.buffer_size

        self.lr = 1e-3
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 100

        self.Ln = 1
        self.action_repeat = 1
        self.max_ep_len = 2990
        # self.buffer_store_len = ceil(self.max_ep_len / self.action_repeat)

        self.save_freq = 1

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name  # Directory for storing trained model
        self.save_interval = int(5e5)

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name


# reward wrapper
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
