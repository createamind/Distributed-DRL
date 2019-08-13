import numpy as np
import tensorflow as tf
import time
import ray
import gym

from hyperparams import HyperParameters
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy
import signal

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# "Pendulum-v0" 'BipedalWalker-v2' 'LunarLanderContinuous-v2'
flags.DEFINE_string("env_name", "LunarLanderContinuous-v2", "game env")
flags.DEFINE_integer("total_epochs", 500, "total_epochs")
flags.DEFINE_integer("num_workers", 1, "number of workers")
flags.DEFINE_integer("num_learners", 1, "number of learners")
flags.DEFINE_string("is_restore", "False", "True or False. True means restore weights from pickle file.")
flags.DEFINE_integer("a_l_ratio", 1, "steps / sample_times")


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
        self.steps, self.sample_times = 0, 0
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.steps += 1

    def sample_batch(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        self.sample_times += 1
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def get_counts(self):
        return self.sample_times, self.steps, self.size


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values, is_restore=False):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.

        if is_restore:
            try:
                pickle_in = open("weights.pickle", "rb")
                self.weights = pickle.load(pickle_in)
                print("****** weights restored! ******")
            except:
                print("------ error: weights.pickle doesn't exist! ------")
        else:
            values = [value.copy() for value in values]
            self.weights = dict(zip(keys, values))
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    # def push(self, keys, values):
    #     for key, value in zip(keys, values):
    #         self.weights[key] += value

    # TODO push gradients or parameters
    def push(self, keys, values):
        values = [value.copy() for value in values]
        for key, value in zip(keys, values):
            self.weights[key] = value

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    # save weights to disk
    def save_weights(self):
        pickle_out = open("weights.pickle","wb")
        pickle.dump(self.weights, pickle_out)
        pickle_out.close()


@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, opt, learner_index):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    agent = Learner(opt, job="learner")
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    # cache for training data and model weights
    print('os.pid:', os.getpid())
    q1 = multiprocessing.Queue(10)
    q2 = multiprocessing.Queue(5)

    def ps_update(q1, q2):
        print('os.pid of put_data():', os.getpid())

        q1.put(copy.deepcopy(ray.get(replay_buffer.sample_batch.remote(opt.batch_size))))

        while True:
            q1.put(copy.deepcopy(ray.get(replay_buffer.sample_batch.remote(opt.batch_size))))

            if not q2.empty():
                keys, values = q2.get()
                ps.push.remote(keys, values)

    p1 = multiprocessing.Process(target=ps_update, args=(q1, q2))
    p1.start()

    def signal_handler(signal, frame):
        p1.terminate()
        exit()

    signal.signal(signal.SIGINT, signal_handler)

    cnt = 1
    while True:
        # batch = ray.get(replay_buffer.sample_batch.remote(opt.batch_size))
        batch = q1.get()
        agent.train(batch)
        if cnt % 300 == 0:
            # print('q1.qsize():', q1.qsize(), 'q2.qsize():', q2.qsize())
            q2.put(agent.get_weights())
            # keys, values = agent.get_weights()
            # ps.push.remote(copy.deepcopy(keys), copy.deepcopy(values))
        cnt += 1

    p1.join()

    #### debug ####
    # # while True:
    # time_1 = time.time()
    # for _ in range(1000):
    #     batch = ray.get(replay_buffer.sample_batch.remote(opt.batch_size))
    # time_2 = time.time()
    # for _ in range(1000):
    #     agent.parameter_update(batch)
    # time_3 = time.time()
    # for _ in range(1000):
    #     keys, values = agent.get_weights()
    # time_4 = time.time()
    # for _ in range(1000):
    #     ps.push.remote(keys, values)
    # time_5 = time.time()
    # print('batch:', time_2-time_1)
    # print('update:', time_3-time_2)
    # print('get_weights:', time_4-time_3)
    # print('push:', time_5-time_4)
    #
    # time.sleep(10000)


@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    env = gym.make(opt.env_name)

    agent = Actor(opt, job="worker")
    keys = agent.get_weights()[0]

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # epochs = opt.total_epochs // opt.num_workers
    total_steps = opt.steps_per_epoch * opt.total_epochs

    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    # TODO opt.start_steps
    for t in range(total_steps):

        if t > opt.start_steps:
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
        d = False if ep_len == opt.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store.remote(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of episode. Training (ep_len times).
        if d or (ep_len == opt.max_ep_len):
            sample_times, steps, _ = ray.get(replay_buffer.get_counts.remote())

            while sample_times > 0 and steps / sample_times > opt.a_l_ratio:
                sample_times, steps, _ = ray.get(replay_buffer.get_counts.remote())
                time.sleep(0.1)

            # update parameters every episode
            weights = ray.get(ps.pull.remote(keys))
            agent.set_weights(keys, weights)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


@ray.remote
def worker_test(ps, replay_buffer, opt, worker_index=0):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    agent = Actor(opt, job="main")

    keys, weights = agent.get_weights()

    # Keep the main process running! Otherwise everything will shut down when main process finished.
    start_time = time.time()
    time0 = time1 = time.time()
    sample_times1, steps, size = ray.get(replay_buffer.get_counts.remote())
    max_ret = -1000

    while True:
        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)

        ep_ret = agent.test(start_time, replay_buffer)
        sample_times2, steps, size = ray.get(replay_buffer.get_counts.remote())
        time2 = time.time()
        print("test_reward:", ep_ret, "sample_times:", sample_times2, "steps:", steps, "buffer_size:", size)
        print('update frequency:', (sample_times2-sample_times1)/(time2-time1), 'total time:', time2 - time0)

        if ep_ret > max_ret:
            ps.save_weights.remote()
            print("****** weights saved! ******")
            max_ret = ep_ret

        time1 = time2
        sample_times1 = sample_times2

        if steps >= opt.total_epochs * opt.steps_per_epoch:
            exit(0)
        # if time2 - time0 > 1500:
        #     exit(0)

        time.sleep(5)


if __name__ == '__main__':

    ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)

    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    # print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    opt = HyperParameters(FLAGS.env_name, FLAGS.total_epochs, FLAGS.num_workers, FLAGS.a_l_ratio)

    # Create a parameter server with some random weights.
    if FLAGS.is_restore == "True":
        ps = ParameterServer.remote([], [], is_restore=True)
    else:
        net = Learner(opt, job="main")
        all_keys, all_values = net.get_weights()
        ps = ParameterServer.remote(all_keys, all_values)

    replay_buffer = ReplayBuffer.remote(obs_dim=opt.obs_dim, act_dim=opt.act_dim, size=opt.replay_size)

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    time.sleep(5)

    task_train = [worker_train.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_learners)]

    task_test = worker_test.remote(ps, replay_buffer, opt)

    ray.wait([task_test, ])