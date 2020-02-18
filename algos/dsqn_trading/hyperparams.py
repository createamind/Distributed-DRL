import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gym
from math import ceil
# sys.path.append('/home/zdx/trading-game/game')
# print(__file__)
from tradingenv import TradingEnv


class HyperParameters:
    def __init__(self, env_name, exp_name, num_workers, a_l_ratio, weights_file):
        # parameters set
        print(__file__)
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

        env_gym = TradingEnv()

        # env = FootballWrapper(env_football)
        env = env_gym

        # self.obs_space = Box(low=-1.0, high=1.0, shape=self.obs_dim, dtype=np.float32)
        self.obs_dim = env.observation_space.shape
        self.obs_space = env.observation_space
        self.obs_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space
        self.act_shape = self.act_space.shape

        print(self.obs_dim)

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
        self.max_ep_len = 100
        # self.buffer_store_len = ceil(self.max_ep_len / self.action_repeat)

        self.save_freq = 1

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name  # Directory for storing trained model
        self.save_interval = int(5e5)

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name
