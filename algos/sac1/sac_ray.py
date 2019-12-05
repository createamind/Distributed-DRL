import numpy as np
import tensorflow as tf
import time
import ray
import gym

from hyperparams import HyperParameters, Wrapper
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy

from collections import deque

import inspect
import json
from ray.rllib.utils.compression import pack, unpack


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string("env_name", "BipedalWalker-v2", "game env")
flags.DEFINE_string("exp_name", "Exp1", "experiments name")
flags.DEFINE_integer("num_workers", 16, "number of workers")
flags.DEFINE_string("weights_file", "", "empty means False. "
                                        "[Maxret_weights.pickle] means restore weights from this pickle file.")
flags.DEFINE_float("a_l_ratio", 2, "steps / sample_times")


@ray.remote(num_cpus=2)
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SQN_N_STEP agents.
    """

    def __init__(self, opt):
        self.opt = opt
        if opt.obs_shape != (115,):
            self.buffer_o = np.array([['0' * 2000] * (opt.Ln + 1)] * opt.buffer_size, dtype=np.str)
        else:
            self.buffer_o = np.zeros((opt.buffer_size, opt.Ln + 1) + opt.obs_shape, dtype=np.float32)
        self.buffer_a = np.zeros((opt.buffer_size, opt.Ln) + opt.act_shape, dtype=np.float32)
        self.buffer_r = np.zeros((opt.buffer_size, opt.Ln), dtype=np.float32)
        self.buffer_d = np.zeros((opt.buffer_size, opt.Ln), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.steps, self.sample_times = 0, 0

    def store(self, o_queue, a_r_d_queue, worker_index):

        obs, = np.stack(o_queue, axis=1)

        if self.opt.obs_shape != (115,):
            self.buffer_o[self.ptr] = obs
        else:
            self.buffer_o[self.ptr] = np.array(list(obs), dtype=np.float32)

        a, r, d, = np.stack(a_r_d_queue, axis=1)
        self.buffer_a[self.ptr] = np.array(list(a), dtype=np.float32)
        self.buffer_r[self.ptr] = np.array(list(r), dtype=np.float32)
        self.buffer_d[self.ptr] = np.array(list(d), dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # TODO
        self.steps += 1 * self.opt.num_buffers
        # self.steps += opt.Ln * opt.action_repeat

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        # TODO
        self.sample_times += 1 * self.opt.num_buffers

        return dict(obs=self.buffer_o[idxs],
                    acts=self.buffer_a[idxs],
                    rews=self.buffer_r[idxs],
                    done=self.buffer_d[idxs], )

    def get_counts(self):
        return self.sample_times, self.steps, self.size


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values, weights_file=""):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.

        if weights_file:
            try:
                with open(weights_file, "rb") as pickle_in:
                    self.weights = pickle.load(pickle_in)
                    print("****** weights restored! ******")
            except:
                print("------------------------------------------------")
                print(weights_file)
                print("------ error: weights file doesn't exist! ------")
                exit()
        else:
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


class Cache(object):

    def __init__(self, replay_buffer):
        # cache for training data and model weights
        print('os.pid:', os.getpid())
        self.replay_buffer = replay_buffer
        self.q1 = multiprocessing.Queue(10)
        self.q2 = multiprocessing.Queue(5)
        self.p1 = multiprocessing.Process(target=self.ps_update, args=(self.q1, self.q2, self.replay_buffer))
        self.p1.daemon = True

    def ps_update(self, q1, q2, replay_buffer):
        print('os.pid of put_data():', os.getpid())

        q1.put(copy.deepcopy(ray.get(replay_buffer[np.random.choice(opt.num_buffers, 1)[0]].sample_batch.remote())))

        while True:
            if q1.qsize() < 10:
                q1.put(copy.deepcopy(ray.get(replay_buffer[np.random.choice(opt.num_buffers, 1)[0]].sample_batch.remote())))

            if not q2.empty():
                keys, values = q2.get()
                ps.push.remote(keys, values)

    def start(self):
        self.p1.start()
        self.p1.join(10)

    def end(self):
        self.p1.terminate()

# TODO
@ray.remote(num_cpus=2, num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, opt, learner_index):
    agent = Learner(opt, job="learner")
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    cache = Cache(replay_buffer)

    cache.start()

    cnt = 1
    while True:
        batch = cache.q1.get()
        if opt.model == "cnn":
            batch['obs'] = np.array([[unpack(o) for o in lno] for lno in batch['obs']])
        agent.train(batch, cnt)
        # TODO cnt % 300 == 0 before
        if cnt % 100 == 0:
            cache.q2.put(agent.get_weights())
        cnt += 1


@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):

    agent = Actor(opt, job="worker")
    keys = agent.get_weights()[0]

    filling_steps = 0
    while True:
        # ------ env set up ------
        env = Wrapper(gym.make(opt.env_name), opt.obs_noise, opt.act_noise, opt.reward_scale, 3)
        # ------ env set up end ------

        ################################## deques

        o_queue = deque([], maxlen=opt.Ln + 1)
        a_r_d_queue = deque([], maxlen=opt.Ln)

        ################################## deques

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        ################################## deques reset
        t_queue = 1
        if opt.model == "cnn":
            compressed_o = pack(o)
            o_queue.append((compressed_o,))
        else:
            o_queue.append((o,))

        ################################## deques reset

        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)

        while True:

            # don't need to random sample action if load weights from local.
            if filling_steps > opt.start_steps or opt.weights_file:
                a = agent.get_action(o, deterministic=False)
            else:
                a = env.action_space.sample()
                filling_steps += 1
            # Step the env
            o2, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # d = False if ep_len*opt.action_repeat >= opt.max_ep_len else d

            o = o2

            #################################### deques store

            a_r_d_queue.append((a, r, d,))
            if opt.model == "cnn":
                compressed_o2 = pack(o2)
                o_queue.append((compressed_o2,))
            else:
                o_queue.append((o2,))

            # scheme 1:
            # TODO  and t_queue % 2 == 0: %1 lead to q smaller
            # TODO
            if t_queue >= opt.Ln and t_queue % opt.save_freq == 0:
                replay_buffer[np.random.choice(opt.num_buffers, 1)[0]].store.remote(o_queue, a_r_d_queue, worker_index)

            t_queue += 1

            #################################### deques store

            # End of episode. Training (ep_len times).
            if d or (ep_len * opt.action_repeat >= opt.max_ep_len):
                # TODO
                sample_times, steps, _ = ray.get(replay_buffer[0].get_counts.remote())

                print('rollout_ep_len:', ep_len * opt.action_repeat, 'rollout_ep_ret:', ep_ret)

                if steps > opt.start_steps:
                    # update parameters every episode
                    weights = ray.get(ps.pull.remote(keys))
                    agent.set_weights(keys, weights)

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

                ################################## deques reset
                t_queue = 1
                if opt.model == "cnn":
                    compressed_o = pack(o)
                    o_queue.append((compressed_o,))
                else:
                    o_queue.append((o,))

                ################################## deques reset


@ray.remote
def worker_test(ps, replay_buffer, opt):
    agent = Actor(opt, job="main")

    test_env = Wrapper(gym.make(opt.env_name), opt.obs_noise, opt.act_noise, opt.reward_scale, 3)

    agent.test(ps, replay_buffer, opt, test_env)


if __name__ == '__main__':

    # ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)
    ray.init()

    # ------ HyperParameters ------
    opt = HyperParameters(FLAGS.env_name, FLAGS.exp_name, FLAGS.num_workers, FLAGS.a_l_ratio,
                          FLAGS.weights_file)
    All_Parameters = copy.deepcopy(vars(opt))
    All_Parameters["wrapper"] = inspect.getsource(Wrapper)
    All_Parameters["obs_space"] = ""
    All_Parameters["act_space"] = ""

    try:
        os.makedirs(opt.save_dir)
    except OSError:
        pass
    with open(opt.save_dir + "/" + 'All_Parameters.json', 'w') as fp:
        json.dump(All_Parameters, fp, indent=4, sort_keys=True)

    # ------ end ------

    if FLAGS.weights_file:
        ps = ParameterServer.remote([], [], weights_file=FLAGS.weights_file)
    else:
        net = Learner(opt, job="main")
        all_keys, all_values = net.get_weights()
        ps = ParameterServer.remote(all_keys, all_values)

    # Experience buffer
    # Methods called on different actors can execute in parallel,
    # and methods called on the same actor are executed serially in the order that they are called.
    # we need more buffer for more workers to keep high store speed.
    replay_buffer = [ReplayBuffer.remote(opt) for i in range(opt.num_buffers)]

    # Start some training tasks.
    for i in range(FLAGS.num_workers):
        worker_rollout.remote(ps, replay_buffer, opt, i)
        time.sleep(0.05)
    # task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    if opt.weights_file:
        fill_steps = opt.start_steps / 100
    else:
        fill_steps = opt.start_steps
    # store at least start_steps in buffer before training
    _, steps, _ = ray.get(replay_buffer[0].get_counts.remote())
    while steps < fill_steps:
        _, steps, _ = ray.get(replay_buffer[0].get_counts.remote())
        print('fill steps before learn:', steps)
        time.sleep(1)

    task_train = [worker_train.remote(ps, replay_buffer, opt, i) for i in range(opt.num_learners)]

    time.sleep(10)
    while True:
        task_test = worker_test.remote(ps, replay_buffer, opt)
        ray.wait([task_test, ])
