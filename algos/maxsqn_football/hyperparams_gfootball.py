import numpy as np
import os
import sys
from gym.spaces import Box
import datetime
import gfootball.env as football_env


class HyperParameters:
    def __init__(self, env_name, exp_name, total_epochs, num_workers=1, a_l_ratio=1):
        # parameters set

        # ray_servr_address = ""

        # self.env_name = 'LunarLanderContinuous-v2'   # 'MountainCarContinuous-v0'
        self.env_name = "academy_3_vs_1_with_keeper"
        self.exp_name = str(exp_name)

        self.a_l_ratio = a_l_ratio

        # gpu memory fraction
        self.gpu_fraction = 0.3

        self.ac_kwargs = dict(hidden_sizes=[400, 300])

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

        self.alpha = 0.2

        self.gamma = 0.997
        self.replay_size = 5000000
        # self.lr = 1e-4
        self.lr = 1e-4
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 512
        self.start_steps = 20000
        self.max_ep_len = 190
        self.save_freq = 1

        self.seed = 0

        self.summary_dir = './tboard_ray'  # Directory for storing tensorboard summary results
        self.save_dir = './model_ray'      # Directory for storing trained model
