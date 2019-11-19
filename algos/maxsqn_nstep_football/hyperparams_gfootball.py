import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env
from numbers import Number


class HyperParameters:
    def __init__(self, env_name, exp_name, num_workers, a_l_ratio, weights_file):
        # parameters set

        self.exp_name = exp_name

        self.env_name = env_name
        # "_random", "_d_True", ""
        self.rollout_env_name = self.env_name + ""

        self.with_checkpoints = False

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
        self.gpu_fraction = 0.5

        self.hidden_size = (300, 400, 300)

        env_football = football_env.create_environment(env_name=self.env_name, stacked=self.stacked,
                                                       representation=self.representation, render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # self.obs_dim = scenario_obsdim[self.env_name]
        self.obs_dim = env.observation_space.shape
        self.obs_space = Box(low=-1.0, high=1.0, shape=self.obs_dim, dtype=np.float32)

        self.o_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space
        self.a_shape = self.act_space.shape

        self.num_workers = num_workers
        self.num_learners = 1

        self.Ln = 16
        self.use_max = False
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.5

        self.use_bn = False
        self.c_regularizer = 0.0

        self.gamma = 0.997
        if self.model == 'cnn':
            self.replay_size = int(3e4)
        else:
            self.replay_size = int(3e6)

        self.lr = 5e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 256

        self.action_repeat = 3
        self.reward_scale = 200
        self.max_ep_len = 2900
        # self.max_ep_len = 490
        self.save_freq = 2

        self.max_ret = 0
        self.game_difficulty = 0
        self.threshold_score = 96

        self.epsilon = 0
        self.epsilon_alpha = 7

        self.seed = 0

        cwd = os.getcwd()

        self.summary_dir = cwd + '/tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = cwd + '/' + self.exp_name  # Directory for storing trained model

        self.log_dir = self.summary_dir + "/" + str(datetime.datetime.now()) + "-workers_num:" + \
                       str(self.num_workers) + "%" + str(self.a_l_ratio) + self.env_name + "-" + self.exp_name


# reward wrapper
class FootballWrapper(object):

    def __init__(self, env, action_repeat, reward_scale):
        self._env = env
        self.action_repeat = action_repeat
        self.reward_scale = reward_scale

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self._env.step(action)

            r += reward

            if done:
                return obs, r * self.reward_scale, done, info

        return obs, r * self.reward_scale, done, info
