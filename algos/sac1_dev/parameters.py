import numpy as np
import os
import sys
import gym
import datetime


class ParametersSac1:
    def __init__(self, env_name, total_epochs, num_workers=1):
        # parameters set

        # ray_servr_address = ""

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = env_name
        # BipedalWalker-v2
        # Pendulum-v0
        # self.env_name = 'MountainCarContinuous-v0'

        # TODO
        env = gym.make(env_name)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = env.action_space.high[0]

        self.ac_kwargs = dict(hidden_sizes=[400, 300])
        # Share information about action space with policy architecture
        self.action_space = env.action_space
        self.ac_kwargs['action_space'] = env.action_space

        self.total_epochs = total_epochs
        self.num_workers = num_workers

        self.alpha = 0.1

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

        self.summary_dir = './tboard_ray_sac1'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ray_sac1'      # Directory for storing trained model
