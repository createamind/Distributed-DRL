import numpy as np
import os
import sys
from gym.spaces import Box
import datetime, copy
import gfootball.env as football_env
from numbers import Number
from math import ceil


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

        self.recover = False
        self.checkpoint_freq = 21600  # 21600s = 6h

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.hidden_size = (600, 800, 600)

        env_football = football_env.create_environment(env_name=self.env_name, stacked=self.stacked,
                                                       representation=self.representation, render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_space = Box(low=-1.0, high=1.0, shape=self.obs_dim, dtype=np.float32)
        self.obs_dim = env.observation_space.shape
        self.obs_space = env.observation_space
        self.obs_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space
        self.act_shape = self.act_space.shape

        self.num_workers = num_workers
        self.num_learners = 1

        self.num_in_pool = 10000  # 3 * num_workers
        self.pool_pop_ratio = 0.333

        self.left_side_ratio = 1.0

        self.right_random = 0.015

        bot = 0.0
        self_pool = 0.2
        ext_pool = 0.0
        self_play = 0.8

        assert bot + self_pool + ext_pool + self_play == 1.0

        self.bot_worker_ratio = bot
        self.self_pool_probability = self_pool/(self_pool+ext_pool+self_play)  # same-weight self-play ratio
        self.ext_pool_probability = ext_pool/(ext_pool+self_play)
        self.pool_push_freq = int(1e4)
        self.a_l_ratio = 20000000

        self.use_max = False
        self.reward_scale = 150
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.use_bn = False
        self.c_regularizer = 0.0

        self.gamma = 0.997
        
        
        self.Ln = 5
        self.action_repeat = 3
        self.max_ep_len = 2990
        self.buffer_store_len = ceil(self.max_ep_len / self.action_repeat)
        
        # self.num_buffers = 1
        self.num_buffers = self.num_workers // 25 + 1
        if self.model == 'cnn':
            self.buffer_size = int(2.5e6) // self.buffer_store_len
        else:
            self.buffer_size = int(2.5e6) // self.buffer_store_len

        self.buffer_size = self.buffer_size // self.num_buffers

        self.start_steps = (int(5e4) // self.buffer_store_len )// self.num_buffers
        if self.weights_file:
            self.start_steps = self.buffer_size

        self.lr = 1e-4
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 5120

        self.save_freq = 1

        self.mu_speed = 7e6
        self.game_difficulty = 1
        self.threshold_score = 96

        self.epsilon = 0
        self.epsilon_alpha = 7

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name  # Directory for storing trained model
        self.save_interval = int(5e5)

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env, action_repeat, reward_scale, right_random=0.0):
        self._env = env
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale
        self.right_random = right_random

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(self.action_repeat):
            
            np.random.seed()
            if np.random.random() < self.right_random:
                act = np.array([action[0], self._env.action_space.sample()[1]])
                # act = np.array([action[0], 0])
            else:
                act = np.array(action)

            obs, reward, done, info = self._env.step(act)

            r += reward

            if obs[0][95] and not obs[0][108]:
                r -= 0.1/self.reward_scale
                print("not normal mode")

            if done:
                return obs, r * self.reward_scale, done, info

        return obs, r * self.reward_scale, done, info
