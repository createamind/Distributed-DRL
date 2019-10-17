import numpy as np
import os
import sys
from gym.spaces import Box
import time
import gfootball.env as football_env


class HyperParameters:
    def __init__(self):
        # parameters set

        self.env_name = "11_vs_11_easy_stochastic"  #'academy_empty_goal' #
        self.rollout_env_name = self.env_name
        self.exp_name = '11v11_easy_86_33_200_done_random'

        self.env_random = True
        self.deterministic = False

        if self.env_random:
            self.rollout_env_name = self.env_name + "_random"
        if self.deterministic:
            self.rollout_env_name = self.env_name + "_d_True"

        self.save_dir = './data/' + self.exp_name    # Directory for storing trained model
        self.is_restore = False


        # gpu memory fraction
        self.gpu_fraction = 0.2

        # self.ac_kwargs = dict(hidden_sizes=[600, 800, 600, 400, 300])
        self.hidden_size = (800, 600)
        # self.hidden_size = (400, 300)

        env_football = football_env.create_environment(env_name=self.env_name, representation='simple115', render=False)

        # env = FootballWrapper(env_football)
        env = env_football

        # # gym env
        # obs_dim = env.observation_space.shape[0]
        # obs_space = env.observation_space

        # google football
        scenario_obsdim = {'academy_empty_goal': 39, 'academy_empty_goal_random': 39}
        scenario_obsdim['academy_3_vs_1_with_keeper'] = 51
        scenario_obsdim['academy_3_vs_1_with_keeper_random'] = 51
        scenario_obsdim['academy_single_goal_versus_lazy'] = 115
        scenario_obsdim['academy_single_goal_versus_lazy_random'] = 115
        scenario_obsdim['11_vs_11_stochastic'] = 115
        scenario_obsdim['11_vs_11_easy_stochastic'] = 115
        scenario_obsdim['11_vs_11_easy_stochastic_random'] = 115
        scenario_obsdim['11_vs_11_stochastic_random'] = 115

        self.obs_dim = scenario_obsdim[self.rollout_env_name]-7
        self.obs_space = Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.o_shape = self.obs_space.shape

        self.act_dim = env.action_space.n
        self.act_space = env.action_space
        self.a_shape = self.act_space.shape


        self.total_epochs = 200000

        self.num_learners = 1
        self.num_workers = 9
        self.a_l_ratio = 0.7


        self.Ln = 3
        self.use_max = False
        self.alpha = 0.1
        # self.alpha = "auto"
        self.target_entropy = 0.4


        self.gamma = 0.997
        self.replay_size = int(2e6)

        self.use_bn = False
        self.c_regularizer = 0.0
        self.lr = 4e-5
        self.polyak = 0.995

        self.steps_per_epoch = 5000
        self.batch_size = 300

        if self.is_restore:
            self.start_steps = int(1e6)
        else:
            self.start_steps = int(3e4)

        self.start_steps_per_worker = int(self.start_steps/self.num_workers)
        self.max_ep_len = 1000
        self.save_freq = 1

        self.seed = int(time.time())

        # Directory for storing tensorboard summary results
        self.summary_dir = './tensorboard_events/'+time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())+ self.env_name + "-" \
                           + self.exp_name + "-workers_num:" + str(self.num_workers) + "%" + str(self.a_l_ratio)



# reward wrapper
class FootballWrapper(object):

    def __init__(self, env):
        self._env = env
        # self.dis_to_goal = 0.0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        # self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.0, 0.0])
        return obs

    def step(self, action):
        r = 0.0
        for _ in range(3):
            obs, reward, done, info = self._env.step(action)
            # if reward != 0.0:
            #     done = True
            # else:
            #     done = False

            if reward < 0.0:
                done = True
                reward = 0.0


            # if reward < 0.0:
            #     reward = 0.0
            # reward -= 0.00175

            # if obs[0] < 0.0:
            #     done = True

            # if not done:  # when env is done, ball position will be reset.
            #     reward += self.incentive(obs)

            r += reward

            if done:
                return obs, r * 200, done, info

        return obs, r * 200, done, info

    def incentive(self, obs):
        # total accumulative incentive reward is around 0.5
        dis_to_goal_new = np.linalg.norm(obs[0:2] - [1.01, 0.0])
        r = 0.25 * (self.dis_to_goal - dis_to_goal_new)
        self.dis_to_goal = dis_to_goal_new
        return r

    def incentive1(self, obs):
        r = -self.dis_to_goal * (1e-4)  # punishment weighted by dis_to_goal
        self.dis_to_goal = np.linalg.norm(obs[0:2] - [1.01, 0.0])  # interval: 0.0 ~ 2.0
        return r