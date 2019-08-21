import numpy as np
import os
import sys
import gym
import datetime


class HyperParameters:
    def __init__(self, env_name, total_epochs, num_workers=1, a_l_ratio=1):
        # parameters set

        # ray_servr_address = ""

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = env_name
        # BipedalWalker-v2
        # Pendulum-v0
        # self.env_name = 'MountainCarContinuous-v0'

        self.a_l_ratio = a_l_ratio

        # self.wrapper = True

        # gpu memory fraction
        self.gpu_fraction = 0.05

        env = gym.make(env_name)
        self.obs_dim = env.observation_space.shape[0]
        self.obs_space = env.observation_space
        self.act_dim = env.action_space.n
        self.act_space = env.action_space

        self.ac_kwargs = dict(hidden_sizes=[400, 300])
        # Share information about action space with policy architecture
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.alpha = 0.2
        # self.alpha = 'auto'

        self.gamma = 0.99
        self.replay_size = 1000000
        self.lr = 1e-3
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 100
        self.start_steps = 10000
        self.max_ep_len = 1000
        self.save_freq = 1

        self.seed = 0

        self.act_noise = 0.3
        self.obs_noise = 0.0
        self.reward_scale = 5.0

        self.summary_dir = './tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ray'      # Directory for storing trained model


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
