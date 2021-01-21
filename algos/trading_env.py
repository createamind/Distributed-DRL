import numpy as np
import gym
from gym import spaces
from gym.spaces import Box
import ctypes
import json
import os
import sys
from collections import deque
import pandas as pd
import pickle
import time

info_names = [
    "Done", "LastPrice", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2",
    "AskPrice2", "AskVolume2", "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3", "BidPrice4",
    "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5", "Volume",
    "HighestPrice", "LowestPrice", "TradingDay", "Target_Num", "Actual_Num", "AliveBidPrice1",
    "AliveBidVolume1", "AliveBidPrice2", "AliveBidVolume2", "AliveBidPrice3", "AliveBidVolume3",
    "AliveAskPrice1", "AliveAskVolume1", "AliveAskPrice2", "AliveAskVolume2", "AliveAskPrice3",
    "AliveAskVolume3", "score", "profit", "total_profit", "baseline_profit", "action", "designed_reward"
]

data_v19_len = [
    225013, 225015, 225015, 225015, 225015, 225017, 225015, 225015, 225017, 225015, 225015, 225015, 225015, 225015,
    225015, 225015, 225015, 225015, 225015, 225015, 225015, 225015, 225010, 225015, 225015, 135002, 225015, 225015,
    225015, 225015, 225015, 225017, 225015, 225017, 225015, 225017, 225015, 225015, 225015, 225015, 225017, 225015,
    225015, 225015, 225017, 225017, 225016, 225017, 225015, 225013, 225015, 225015, 225017, 225017, 225014, 225017,
    225015, 225013, 225015, 225017, 225015, 225015, 225015, 225017, 225015, 225017, 225017, 225015, 225015, 225015,
    225017, 225017, 225015, 225015, 225017, 225015, 225015, 225017, 225015, 225015, 225014, 225015, 225015, 225015,
    225015, 225015, 225017, 225017, 225015, 225015, 225015, 225015, 225017, 225015, 225017, 225015, 225015, 225015,
    225015, 99005, 225015, 225017, 99009, 225015, 225015, 225009, 225017, 225015, 225015, 225015, 225013, 225013,
    225015, 225015, 225013, 225015, 225015, 225017, 225015, 126016
]   # 120days


class TradingEnv(gym.Env):

    def __init__(self, action_scheme_id, obs_dim, auto_follow=0, max_ep_len=3000, render=False):
        super(TradingEnv, self).__init__()

        self.data_len = data_v19_len
        self.trainning_set = 90
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(ROOT_DIR + "/rl_game/game/")
        so_file = "./game.so"
        self.expso = ctypes.cdll.LoadLibrary(so_file)
        arr_len = 100
        arr1 = ctypes.c_int * arr_len
        arr = ctypes.c_int * 1

        self.ctx = None

        self.actions = arr1()
        self.action_len = arr()
        self.raw_obs = arr1()
        self.raw_obs_len = arr()
        self.rewards = arr1()
        self.rewards_len = arr()

        self._step = self._action_schemes(action_scheme_id)
        self.auto_follow = auto_follow

        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.step_len = 0

        self.max_ep_len = max_ep_len
        self.render = render

        self.his_price = deque(maxlen=30)

    def reset(self, start_day=None, start_skip=None, burn_in=0):

        # random start_day if no start_day
        if start_day is None:
            start_day = np.random.randint(1, self.trainning_set + 1, 1)[0]  # first self.trainning_set days

        # random start_skip if no start_skip
        if start_skip is None:
            day_index = start_day - 1
            max_point = self.data_len[day_index] - self.max_ep_len - burn_in - 50
            start_skip = int(np.random.randint(0, max_point, 1)[0])

        start_info = {"date_index": "{} - {}".format(start_day, start_day), "skip_steps": start_skip}
        # print(start_info)
        if self.ctx:
            self.close_env()
        self.ctx = self.expso.CreateContext(json.dumps(start_info).encode())
        self.expso.GetActions(self.ctx, self.actions, self.action_len)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)

        self.step_len = 0

        obs = self._get_obs(self.raw_obs)

        if self.render:
            self.rendering()

        return obs

    def step(self, action):
        target_num = self.raw_obs[26]
        actual_num = self.raw_obs[27]

        if self.auto_follow is not 0:
            if abs(actual_num - target_num) > self.auto_follow:
                if target_num > actual_num:
                    action = 6
                else:
                    action = 9

        self._step(action)
        self.expso.Step(self.ctx)
        self.expso.GetInfo(self.ctx, self.raw_obs, self.raw_obs_len)
        self.expso.GetReward(self.ctx, self.rewards, self.rewards_len)

        self.step_len += 1

        target_bias = abs(self.raw_obs[27] - self.raw_obs[26])

        obs = self._get_obs(self.raw_obs)
        reward = -target_bias
        done = bool(self.raw_obs[0]) or self.max_ep_len == self.step_len

        info = {"TradingDay": self.raw_obs[25],
                "score": self.rewards[0],
                "profit": self.rewards[1],
                "target_bias": target_bias}

        if self.render and self.obs_dim == 38:
            self.rendering(action)

        self.his_price.append(obs[0])
        obs[22] = max(self.his_price)
        obs[23] = min(self.his_price)

        return obs, reward, done, info

    def _get_obs(self, raw_obs):

        price_mean = 26440.28
        price_max = 27952.0
        bid_ask_volume_log_mean = 1.97
        bid_ask_volume_log_max = 6.42
        total_volume_mean = 120755.66
        total_volume_max = 321988.0
        # target_abs_mean = 51.018
        target_mean = 2.55
        target_max = 311.0

        price_filter = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 24, 28, 30, 32, 36, 38, 40]
        bid_ask_volume_filter = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 31, 33, 37, 39, 41]
        total_volume_filter = [22]
        target_filter = [26, 27]
        obs = np.array(raw_obs[:44], dtype=np.float32)

        obs[price_filter] = (obs[price_filter] - price_mean) / (price_max - price_mean)
        obs[bid_ask_volume_filter] = (np.log(obs[bid_ask_volume_filter]) - bid_ask_volume_log_mean) / (
                bid_ask_volume_log_max - bid_ask_volume_log_mean)
        obs[total_volume_filter] = (obs[total_volume_filter] - total_volume_mean) / (
                total_volume_max - total_volume_mean)
        obs[target_filter] = (obs[target_filter] - target_mean) / (target_max - target_mean)

        if self.obs_dim == 38:
            obs = np.delete(obs, [0, 25, 34, 35, 42, 43])
        elif self.obs_dim == 26:
            obs = obs[:28]
            obs = np.delete(obs, [0, 25])
        elif self.obs_dim == 24:
            obs = obs[:25]
            obs = np.delete(obs, [0])
        else:
            assert False, "incorrect obs_dim!"
        obs[obs < -1] = -1
        obs[obs > 1] = 1

        return obs

    def _action_schemes(self, action_scheme_id):

        schemes = {}

        def scheme3(action):
            assert 0 <= action <= 2 or action == 6 or action == 9, "action should be 0,1,2"
            if action == 1:
                self.expso.Action(self.ctx, self.actions[18])  # 如果是买动作，卖方向全撤。
                self.expso.Action(self.ctx, self.actions[6])
            elif action == 2:
                self.expso.Action(self.ctx, self.actions[15])  # 如果是卖动作，买方向全撤。
                self.expso.Action(self.ctx, self.actions[9])
            elif action == 0:
                self.expso.Action(self.ctx, self.actions[action])
            # for auto_clip
            elif action == 6:
                self.expso.Action(self.ctx, self.actions[18])
                self.expso.Action(self.ctx, self.actions[6])
            elif action == 9:
                self.expso.Action(self.ctx, self.actions[15])
                self.expso.Action(self.ctx, self.actions[9])

        schemes[3] = scheme3

        # 根据买卖方向进行自动反方向撤单操作
        def scheme15(action):
            assert 0 <= action <= 14, "action should be 0,1,...,14"
            if 1 <= action <= 7:
                self.expso.Action(self.ctx, self.actions[18])  # 如果是买动作，卖方向全撤。
            elif 8 <= action <= 14:
                self.expso.Action(self.ctx, self.actions[15])  # 如果是卖动作，买方向全撤。
            # 执行action
            self.expso.Action(self.ctx, self.actions[action])

        schemes[15] = scheme15

        # 学习全撤单操作
        def scheme17(action):
            assert 0 <= action <= 16, "action should <=16"
            if action <= 14:
                self.expso.Action(self.ctx, self.actions[action])
            elif action == 15:
                self.expso.Action(self.ctx, self.actions[15])
            elif action == 16:
                self.expso.Action(self.ctx, self.actions[18])

        schemes[17] = scheme17

        # 全部操作
        def scheme21(action):
            assert 0 <= action <= 20, "action should be 0,1,...,20"
            self.expso.Action(self.ctx, self.actions[action])

        schemes[21] = scheme21

        # 这里添加新的scheme...
        # def scheme0(action):
        #     pass
        # schemes[0] = scheme0

        self.action_dim = action_scheme_id
        self.action_space = spaces.Discrete(self.action_dim)

        return schemes[action_scheme_id]

    def policy_069(self):  # actions: 0,6,9
        if self.raw_obs[26] > self.raw_obs[27]:
            action = 6
        elif self.raw_obs[26] < self.raw_obs[27]:
            action = 9
        else:
            action = 0
        return action

    def rendering(self, action=None):
        print("-----------------------")
        print("Action:", action)
        print("AliveAskPriceNUM:", self.raw_obs[42])
        print("AliveAskVolumeNUM:", self.raw_obs[43])
        print("AliveAskPrice3:", self.raw_obs[40])
        print("AliveAskVolume3:", self.raw_obs[41])
        print("AliveAskPrice2:", self.raw_obs[38])
        print("AliveAskVolume2:", self.raw_obs[39])
        print("AliveAskPrice1:", self.raw_obs[36])
        print("AliveAskVolume1:", self.raw_obs[37])
        print("AskPrice1:", self.raw_obs[4])
        print("AskVolume1:", self.raw_obs[5])
        print(".....")
        print("LastPrice:", self.raw_obs[1])
        print("Actual_Num:", self.raw_obs[27])
        print(".....")
        print("BidPrice1:", self.raw_obs[2])
        print("BidVolume1:", self.raw_obs[3])
        print("AliveBidPrice1:", self.raw_obs[28])
        print("AliveBidVolume1:", self.raw_obs[29])
        print("AliveBidPrice2:", self.raw_obs[30])
        print("AliveBidVolume2:", self.raw_obs[31])
        print("AliveBidPrice3:", self.raw_obs[32])
        print("AliveBidVolume3:", self.raw_obs[33])
        print("AliveBidPriceNUM:", self.raw_obs[34])
        print("AliveBidVolumeNUM:", self.raw_obs[35])
        print("-----------------------")

    def close_env(self):
        self.expso.ReleaseContext(self.ctx)


class FrameStack(gym.Wrapper):
    def __init__(self, env, frame_stack, jump=1, model='mlp'):
        super(FrameStack, self).__init__(env)
        self.frame_stack = frame_stack
        self.jump = jump
        self.model = model
        self.total_frame = frame_stack * jump
        self.frames = deque([], maxlen=self.total_frame)
        if model == 'mlp':
            self.obs_dim = self.env.observation_space.shape[0] * frame_stack
            self.observation_space = Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.observation_space = Box(-np.inf, np.inf, shape=(frame_stack, self.env.observation_space.shape[0]),
                                         dtype=np.float32)

    def reset(self, start_day=None, start_skip=None, duration=None, burn_in=0):
        ob = self.env.reset(start_day=start_day, start_skip=start_skip, duration=duration, burn_in=burn_in)
        ob = np.float32(ob)
        for _ in range(self.total_frame):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = np.float32(ob)
        self.frames.append(ob)
        return self.observation(), reward, done, info

    def observation(self):
        assert len(self.frames) == self.total_frame
        obs_stack = np.array(self.frames)
        idx = np.arange(0, self.total_frame, self.jump)
        obs = obs_stack[idx]
        if self.model == 'mlp':
            return np.stack(obs, axis=0).reshape((self.obs_dim,))
        else:
            return obs


if __name__ == "__main__":

    env = TradingEnv(action_scheme_id=3, obs_dim=38)
    # env = FrameStack(env, frame_stack=3, jump=3, model='cnn')

    cnt = 0

    for i in range(1):

        obs = env.reset()

        # burn-in
        # while env.target_diffs < 50:
        #     action = env.baseline_policy(obs)
        #     obs, reward, done, info = env.step(action)
        #     cnt += 1
        # print("burn-in steps:", cnt)

        # print(env.raw_obs[26], env.raw_obs[27])
        print(obs)
        step = 1
        t0 = time.time()
        price = 0.0
        while True:
            action = env.action_space.sample()
            # action = env.baseline_policy(obs)
            # action = 0
            obs, reward, done, info = env.step(action)
            step += 1
            if step % 10 == 0:
                # print(step, env.raw_obs[26], env.raw_obs[27],
                #       (info["profit"], info["total_profit"], info["baseline_profit"]),
                #       (info["baseline_profit"] - info["profit"]) * 10 / info["target_diffs"], info["score"],
                #       (info["reward_score"], info["reward_target"], info["reward_action"],))
                print(obs)
            # if price != info["price"]:
            #     print('='*66)
            #     price = info["price"]
            if done:
                print("Done!", done, cnt, step, 'time:', time.time() - t0)
                # all_data = env.all_data
                # all_data_df = pd.DataFrame(all_data)
                # print(all_data_df.tail())
                break

    env.close_env()
