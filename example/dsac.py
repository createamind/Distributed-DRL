import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger

import ray
import pickle
from model import Model


@ray.remote
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.rollout_steps = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.rollout_steps += 1

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def get_counts(self):
        return self.rollout_steps


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        values = [value.copy() for value in values]
        for key, value in zip(keys, values):
            self.weights[key] = value

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def get_weights(self):
        return self.weights

    # save weights to disk
    def save_weights(self, name):
        with open(name + "weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)


@ray.remote
def worker_rollout(ps, replay_buffer, args):
    env = gym.make(args.env)
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = args.steps_per_epoch * args.epochs

    agent = Model(args)
    keys = agent.get_weights()[0]

    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > args.start_steps:
            a = agent.get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == args.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store.remote(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == args.max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """

            # print(ep_len, ep_ret)
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            weights = ray.get(ps.pull.remote(keys))
            agent.set_weights(keys, weights)


@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, args):
    agent = Model(args)
    keys = agent.get_weights()[0]

    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    cnt = 1
    while True:

        agent.train(replay_buffer, args)

        if cnt % 300 == 0:
            keys, values = agent.get_weights()
            ps.push.remote(keys, values)

        cnt += 1


@ray.remote
def worker_test(ps, start_time):

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    config = locals()
    del config['ps']
    logger.save_config(config)

    agent = Model(args)
    keys = agent.get_weights()[0]

    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)
    test_env = gym.make(args.env)
    while True:
        ave_ret = agent.test_agent(test_env, args)
        # print("test Average Ret:", ave_ret, "time:", time.time()-start_time)
        logger.log_tabular('AverageTestEpRet', ave_ret)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--exp_name', type=str, default='dsac_6worker_E1')
    args = parser.parse_args()

    # ac_kwargs = dict()
    args.seed = 0
    args.steps_per_epoch = 5000
    args.epochs = 100
    args.replay_size = int(1e6)
    args.gamma = 0.99,
    args.polyak = 0.995
    args.lr = 1e-3
    args.alpha = 0.2
    args.batch_size = 100
    args.start_steps = 10000
    args.max_ep_len = 1000
    args.logger_kwargs = dict()
    args.save_freq = 1
    args.ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)

    env = gym.make(args.env)
    args.obs_dim = env.observation_space.shape[0]
    args.act_dim = env.action_space.shape[0]
    # Share information about action space with policy architecture
    args.ac_kwargs['action_space'] = env.action_space

    args.num_workers = 6
    args.num_learners = 1

    ray.init()

    net = Model(args)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    replay_buffer = ReplayBuffer.remote(args.obs_dim, args.act_dim, args.replay_size)

    start_time = time.time()

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, args) for i in range(args.num_workers)]

    time.sleep(20)

    task_train = [worker_train.remote(ps, replay_buffer, args) for i in range(args.num_learners)]

    time.sleep(10)

    task_test = worker_test.remote(ps, start_time)
    ray.wait(task_rollout)
