import numpy as np
import tensorflow as tf
import time
import ray
import gym
from collections import deque

from hyperparams import HyperParameters, Wrapper
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy

import inspect
import json
from ray.rllib.utils.compression import pack, unpack


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("env_name", "LunarLander-v2", "game env")
flags.DEFINE_string("exp_name", "Exp1W1", "experiments name")
flags.DEFINE_integer("num_workers", 1, "number of workers")
flags.DEFINE_string("weights_file", "", "empty means False. "
                                        "[Maxret_weights.pickle] means restore weights from this pickle file.")
flags.DEFINE_string("weights_folder_path", "", "empty means False. ")
flags.DEFINE_string("ext_weights_folder_path", "", "empty means False. ")
flags.DEFINE_float("a_l_ratio", 10, "actor_steps / learner_steps")
flags.DEFINE_bool("recover", False, "back training from last checkpoint")
flags.DEFINE_string("checkpoint_path", "", "empty means opt.save_dir. ")


@ray.remote(num_cpus=2)
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SQN_N_STEP agents.
    """

    def __init__(self, opt, buffer_index):
        self.opt = opt
        self.buffer_index = buffer_index
        if opt.model == "cnn":
            self.buffer_o = np.array([['0' * 2000] * (opt.Ln + 1)] * opt.buffer_size, dtype=np.str)
        else:
            self.buffer_o = np.zeros((opt.buffer_size, opt.Ln + 1) + opt.obs_shape, dtype=np.float32)
        self.buffer_a = np.zeros((opt.buffer_size, opt.Ln) + opt.act_shape, dtype=np.float32)
        self.buffer_r = np.zeros((opt.buffer_size, opt.Ln), dtype=np.float32)
        self.buffer_d = np.zeros((opt.buffer_size, opt.Ln), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.actor_steps, self.learner_steps = 0, 0

    def store(self, o_queue, a_r_d_queue, worker_index):

        obs, = np.stack(o_queue, axis=1)

        if opt.model == "cnn":
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
        self.actor_steps += self.opt.num_buffers
        # self.actor_steps += self.buffer_store_len * self.action_repeat * self.opt.num_buffers
        # self.actor_steps += opt.Ln * opt.action_repeat

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        # idxs2 = np.random.randint(0, self.opt.buffer_store_len-self.opt.Ln, size=1)[0]

        self.learner_steps += 1 * self.opt.num_buffers

        # buffer_o shape: (buffer size, max_ep_len, obs)
        # speed up slice using fancy indexing and broadcasting
        # return dict(obs=self.buffer_o[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln + 1)],
        #             acts=self.buffer_a[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)],
        #             rews=self.buffer_r[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)],
        #             done=self.buffer_d[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)], )
        return dict(obs=self.buffer_o[idxs],
                    acts=self.buffer_a[idxs],
                    rews=self.buffer_r[idxs],
                    done=self.buffer_d[idxs], )

    def get_counts(self):
        return self.learner_steps, self.actor_steps, self.size

    def save(self):
        np.save(opt.save_dir + "/checkpoint/" + 'buffer_o-' + str(self.buffer_index), self.buffer_o)
        np.save(opt.save_dir + "/checkpoint/" + 'buffer_a-' + str(self.buffer_index), self.buffer_a)
        np.save(opt.save_dir + "/checkpoint/" + 'buffer_r-' + str(self.buffer_index), self.buffer_r)
        np.save(opt.save_dir + "/checkpoint/" + 'buffer_d-' + str(self.buffer_index), self.buffer_d)
        buffer_counts = np.array((self.ptr, self.size, self.max_size, self.actor_steps, self.learner_steps))
        np.save(opt.save_dir + "/checkpoint/" + 'buffer_counts-' + str(self.buffer_index), buffer_counts)
        print("****** buffer " + str(self.buffer_index) + " saved! ******")

    def load(self, checkpoint_path):
        if not checkpoint_path:
            checkpoint_path = opt.save_dir + "/checkpoint"
        if opt.obs_shape != (115,):
            self.buffer_o = np.load(checkpoint_path + '/buffer_o-' + str(self.buffer_index) + '.npy')
        else:
            self.buffer_o = np.load(checkpoint_path + '/buffer_o-' + str(self.buffer_index) + '.npy')
        self.buffer_a = np.load(checkpoint_path + '/buffer_a-' + str(self.buffer_index) + '.npy')
        self.buffer_r = np.load(checkpoint_path + '/buffer_r-' + str(self.buffer_index) + '.npy')
        self.buffer_d = np.load(checkpoint_path + '/buffer_d-' + str(self.buffer_index) + '.npy')
        buffer_counts = np.load(checkpoint_path + '/buffer_counts-' + str(self.buffer_index) + '.npy')
        self.ptr, self.size, self.max_size, self.actor_steps, self.learner_steps = buffer_counts[0], buffer_counts[1], buffer_counts[2], buffer_counts[3], buffer_counts[4]
        print("****** buffer number " + str(self.buffer_index) + " restored! ******")
        print("****** buffer number " + str(self.buffer_index) + " info:", self.ptr, self.size, self.max_size, self.actor_steps, self.learner_steps)


@ray.remote
class ParameterServer(object):
    def __init__(self, opt, keys, values, weights_file="", checkpoint_path=""):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
        self.opt = opt

        if not checkpoint_path:
            checkpoint_path = opt.save_dir + "/checkpoint"

        if opt.recover:
            with open(checkpoint_path + "/checkpoint_weights.pickle", "rb") as pickle_in:
                self.weights = pickle.load(pickle_in)
                print("****** weights restored! ******")

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

        if not opt.recover and not weights_file:
            values = [value.copy() for value in values]
            self.weights = dict(zip(keys, values))

    def push(self, keys, values):
        values = [value.copy() for value in values]
        for key, value in zip(keys, values):
            self.weights[key] = value

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
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
            # print(q1.qsize())
            if q1.qsize() < 10:
                q1.put(copy.deepcopy(
                    ray.get(replay_buffer[np.random.choice(opt.num_buffers, 1)[0]].sample_batch.remote())))

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
        time1 = time.time()
        batch = cache.q1.get()
        time2 = time.time()
        # print('cache get time:', time2-time1)
        if opt.model == "cnn":
            batch['obs'] = np.array([[unpack(o) for o in lno] for lno in batch['obs']])
        agent.train(batch, cnt)
        time3 = time.time()
        # print('agent train time:', time3 - time2)
        # TODO cnt % 300 == 0 before
        if cnt % 100 == 0:
            cache.q2.put(agent.get_weights())
        cnt += 1


@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):

    agent = Actor(opt, job="worker")
    keys = agent.get_weights()[0]
    np.random.seed()
    rand_buff1 = np.random.choice(opt.num_buffers, 1)[0]

    random_steps = 0

    while True:
        # ------ env set up ------

        env = gym.make(opt.env_name)
        # env = Wrapper(env, opt.action_repeat, opt.reward_scale)
        # ------ env set up end ------

        o_queue = deque([], maxlen=opt.Ln + 1)
        a_r_d_queue = deque([], maxlen=opt.Ln)

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if opt.model == "cnn":
            compressed_o = pack(o)
            o_queue.append((compressed_o,))
        else:
            o_queue.append((o,))

        t_queue = 1

        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)

        # for a_l_ratio control
        np.random.seed()
        rand_buff = np.random.choice(opt.num_buffers, 1)[0]
        last_learner_steps, last_actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

        while True:

            # don't need to random sample action if load weights from local.
            if random_steps > opt.start_steps or opt.weights_file or opt.recover:
                a = agent.get_action(o, deterministic=False)
            else:
                a = env.action_space.sample()
                random_steps += 1
            # Step the env
            o2, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # d = False if ep_len*opt.action_repeat >= opt.max_ep_len else d

            o = o2

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

            # End of episode. Training (ep_len times).
            # if d or (ep_len * opt.action_repeat >= opt.max_ep_len):
            if d:
                sample_times, steps, _ = ray.get(replay_buffer[0].get_counts.remote())

                print('rollout_ep_len:', ep_len * opt.action_repeat, 'rollout_ep_ret:', ep_ret)

                if steps > opt.start_steps:
                    # update parameters every episode
                    weights = ray.get(ps.pull.remote(keys))
                    agent.set_weights(keys, weights)

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


                t_queue = 1
                if opt.model == "cnn":
                    compressed_o = pack(o)
                    o_queue.append((compressed_o,))
                else:
                    o_queue.append((o,))



                # for a_l_ratio control
                learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

                while (actor_steps - last_actor_steps) / (learner_steps - last_learner_steps + 1) > opt.a_l_ratio and last_learner_steps > 0:
                    time.sleep(1)
                    learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())


@ray.remote
def worker_test(ps, replay_buffer, opt):
    agent = Actor(opt, job="main")
    test_env = gym.make(opt.env_name)
    agent.test(ps, replay_buffer, opt, test_env)


if __name__ == '__main__':

    # ray.init(object_store_memory=20000000000, redis_max_memory=5000000000, driver_object_store_memory=6000000000)
    ray.init()

    # ------ HyperParameters ------
    opt = HyperParameters(FLAGS.env_name, FLAGS.exp_name, FLAGS.num_workers, FLAGS.a_l_ratio,
                          FLAGS.weights_file)
    if FLAGS.recover:
        opt.recover = True
    All_Parameters = copy.deepcopy(vars(opt))
    All_Parameters["wrapper"] = inspect.getsource(Wrapper)
    import importlib

    All_Parameters["obs_space"] = ""
    All_Parameters["act_space"] = ""

    try:
        os.makedirs(opt.save_dir)
        os.makedirs(opt.save_dir + '/checkpoint')
    except OSError:
        pass
    with open(opt.save_dir + "/" + 'All_Parameters.json', 'w') as fp:
        json.dump(All_Parameters, fp, indent=4, sort_keys=True)

    # ------ end ------

    if FLAGS.weights_file or FLAGS.recover:
        ps = ParameterServer.remote(opt, [], [], weights_file=FLAGS.weights_file, checkpoint_path=FLAGS.checkpoint_path)
    else:
        net = Learner(opt, job="main")
        all_keys, all_values = net.get_weights()
        ps = ParameterServer.remote(opt, all_keys, all_values)

    # Experience buffer
    # Methods called on different actors can execute in parallel,
    # and methods called on the same actor are executed serially in the order that they are called.
    # we need more buffer for more workers to keep high store speed.
    replay_buffer = [ReplayBuffer.remote(opt, i) for i in range(opt.num_buffers)]

    if FLAGS.recover:
        buffer_load_op = [replay_buffer[i].load.remote(FLAGS.checkpoint_path) for i in range(opt.num_buffers)]
        ray.wait(buffer_load_op, num_returns=opt.num_buffers)

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    if not opt.recover:
        # store at least start_steps in buffer before training
        _, actor_steps, size = ray.get(replay_buffer[0].get_counts.remote())
        while size < opt.start_steps:
            _, actor_steps, size = ray.get(replay_buffer[0].get_counts.remote())
            print('start steps before learning:', size, '/', opt.start_steps)
            time.sleep(1)
    else:
        time.sleep(3)

    task_train = [worker_train.remote(ps, replay_buffer, opt, i) for i in range(opt.num_learners)]

    time.sleep(10)
    while True:
        task_test = worker_test.remote(ps, replay_buffer, opt)
        ray.wait([task_test, ])
