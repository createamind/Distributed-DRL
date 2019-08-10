import numpy as np
import tensorflow as tf
import time
import ray
import gym

from parameters import ParametersSac1
import sac1_model
import os


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("env_name", 'BipedalWalker-v2', "game env")  # "Pendulum-v0"
flags.DEFINE_integer("total_epochs", 500, "total_epochs")
flags.DEFINE_integer("num_workers", 1, "number of workers")
flags.DEFINE_integer("num_learners", 1, "number of learners")


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

    def sample_batch(self, batch_size=32):
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
    def __init__(self, keys, values):
        # These values will be mutated, so we must create a copy that is not
        # backed by the object store.
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


@ray.remote(num_gpus=1, max_calls=1)
def learner_task(ps, replay_buffer, opt, learner_index):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    net = sac1_model.Sac1(opt, job="learner")
    keys = net.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    net.set_weights(keys, weights)

    while True:
        batch = ray.get(replay_buffer.sample_batch.remote(opt.batch_size))
        net.parameter_update(batch)
        keys, values = net.get_weights()
        ps.push.remote(keys, values)


@ray.remote
def worker_task(ps, replay_buffer, opt, worker_index):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    env = gym.make(opt.env_name)

    net = sac1_model.Sac1(opt, job="worker")
    keys = net.get_weights()[0]

    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # epochs = opt.total_epochs // opt.num_workers
    total_steps = opt.steps_per_epoch * opt.total_epochs

    weights = ray.get(ps.pull.remote(keys))
    net.set_weights(keys, weights)

    # TODO opt.start_steps
    for t in range(total_steps):

        if t > opt.start_steps:
            a = net.get_action(o)
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
            while sample_times > 0 and steps / sample_times > 2:
                sample_times, steps, _ = ray.get(replay_buffer.get_counts.remote())
                time.sleep(0.1)

            # update parameters every episode
            weights = ray.get(ps.pull.remote(keys))
            net.set_weights(keys, weights)

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


if __name__ == '__main__':

    ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)

    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    # print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    opt = ParametersSac1(FLAGS.env_name, FLAGS.total_epochs, FLAGS.num_workers)

    # Create a parameter server with some random weights.
    net = sac1_model.Sac1(opt, job="main")

    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)
    replay_buffer = ReplayBuffer.remote(obs_dim=opt.obs_dim, act_dim=opt.act_dim, size=opt.replay_size)

    # Start some training tasks.
    worker_tasks = [worker_task.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    time.sleep(5)

    start_time = time.time()
    learner_task = [learner_task.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_learners)]

    # Keep the main process running! Otherwise everything will shut down when main process finished.
    time1 = time.time()
    while True:
        weights = ray.get(ps.pull.remote(all_keys))
        net.set_weights(all_keys, weights)

        ep_ret = net.test_agent(start_time, replay_buffer)
        sample_times, steps, size = ray.get(replay_buffer.get_counts.remote())
        time2 = time.time()
        print("test_reward:", ep_ret, "sample_times:", sample_times, "steps:", steps, "buffer_size:", size)
        print('update frequency:', sample_times/(time2-time1))
        if steps >= opt.total_epochs * opt.steps_per_epoch:
            exit(0)
        time.sleep(5)
