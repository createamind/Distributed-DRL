import numpy as np
import tensorflow as tf
import gym
import time
import ray

import os
import sys

from hyperparams import HyperParameters
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from trading_env import TradingEnv, FrameStack


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("env_name", "Trading", "game env")
flags.DEFINE_string("exp_name", "ddqn-trading", "experiments name")
flags.DEFINE_integer("num_nodes", 1, "number of nodes")
flags.DEFINE_integer("num_workers", 6, "number of workers")
flags.DEFINE_string("weights_file", "", "empty means False.")
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
        self.obs1_buf = np.zeros([opt.buffer_size, opt.obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([opt.buffer_size, opt.obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(opt.buffer_size, dtype=np.float32)
        self.rews_buf = np.zeros(opt.buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(opt.buffer_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.actor_steps, self.learner_steps = 0, 0

    def store(self, obs, act, rew, next_obs, done, worker_index):

        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.actor_steps += 1

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        self.learner_steps += 1
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def get_counts(self):
        return self.learner_steps, self.actor_steps, self.size

    # debug
    def show(self):
        return self.obs1_buf, self.ptr, self.size, self.max_size

    def save(self):
        np.save(self.opt.save_dir + "/checkpoint/" + 'obs1_buf-' + str(self.buffer_index), self.obs1_buf)
        np.save(self.opt.save_dir + "/checkpoint/" + 'obs2_buf-' + str(self.buffer_index), self.obs2_buf)
        np.save(self.opt.save_dir + "/checkpoint/" + 'acts_buf-' + str(self.buffer_index), self.acts_buf)
        np.save(self.opt.save_dir + "/checkpoint/" + 'rews_buf-' + str(self.buffer_index), self.rews_buf)
        np.save(self.opt.save_dir + "/checkpoint/" + 'done_buf-' + str(self.buffer_index), self.done_buf)
        buffer_infos = np.array((self.ptr, self.size, self.max_size, self.actor_steps, self.learner_steps))
        np.save(self.opt.save_dir + "/checkpoint/" + 'buffer_infos-' + str(self.buffer_index), buffer_infos)
        print("****** buffer " + str(self.buffer_index) + " saved! ******")

    def load(self, checkpoint_path):
        if not checkpoint_path:
            checkpoint_path = self.opt.save_dir + "/checkpoint"

        self.obs1_buf = np.load(checkpoint_path + '/obs1_buf-' + str(self.buffer_index) + '.npy')
        self.obs2_buf = np.load(checkpoint_path + '/obs2_buf-' + str(self.buffer_index) + '.npy')
        self.acts_buf = np.load(checkpoint_path + '/acts_buf-' + str(self.buffer_index) + '.npy')
        self.rews_buf = np.load(checkpoint_path + '/rews_buf-' + str(self.buffer_index) + '.npy')
        self.done_buf = np.load(checkpoint_path + '/done_buf-' + str(self.buffer_index) + '.npy')
        buffer_infos = np.load(checkpoint_path + '/buffer_infos-' + str(self.buffer_index) + '.npy')

        self.ptr, self.size, self.max_size, self.actor_steps, self.learner_steps = buffer_infos[0], buffer_infos[1], \
                                                                                   buffer_infos[2], buffer_infos[3], \
                                                                                   buffer_infos[4]
        print("****** buffer number " + str(self.buffer_index) + " restored! ******")
        print("****** buffer number " + str(self.buffer_index) + " infos:", self.ptr, self.size, self.max_size,
              self.actor_steps, self.learner_steps)


@ray.remote(num_cpus=2)
class ParameterServer:
    def __init__(self, opt, weights_file, checkpoint_path, ps_index):
        # each node will have a Parameter Server

        self.opt = opt
        self.learner_step = 0
        net = Learner(opt, job="ps")
        keys, values = net.get_weights()

        # --- make dir for all nodes and save parameters ---
        try:
            os.makedirs(opt.save_dir)
            os.makedirs(opt.save_dir + '/checkpoint')
        except OSError:
            pass
        all_parameters = copy.deepcopy(vars(opt))
        all_parameters["obs_space"] = ""
        all_parameters["act_space"] = ""
        with open(opt.save_dir + "/" + 'All_Parameters.json', 'w') as fp:
            json.dump(all_parameters, fp, indent=4, sort_keys=True)
        # --- end ---

        self.weights = None

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
        self.learner_step += opt.push_freq

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(self.opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)


class Cache(object):

    def __init__(self, node_buffer):
        # cache for training data and model weights
        print('os.pid:', os.getpid())
        self.node_buffer = node_buffer
        self.q1 = multiprocessing.Queue(12)
        self.q2 = multiprocessing.Queue(5)
        self.p1 = multiprocessing.Process(target=self.ps_update, args=(self.q1, self.q2, self.node_buffer))
        self.p1.daemon = True

    def ps_update(self, q1, q2, node_buffer):
        print('os.pid of put_data():', os.getpid())

        node_idx = np.random.choice(opt.num_nodes, 1)[0]
        buffer_idx = np.random.choice(opt.num_buffers, 1)[0]
        q1.put(copy.deepcopy(ray.get(node_buffer[node_idx][buffer_idx].sample_batch.remote())))

        while True:
            if q1.qsize() < 10:
                node_idx = np.random.choice(opt.num_nodes, 1)[0]
                buffer_idx = np.random.choice(opt.num_buffers, 1)[0]
                q1.put(copy.deepcopy(ray.get(node_buffer[node_idx][buffer_idx].sample_batch.remote())))

            if not q2.empty():
                keys, values = q2.get()
                [node_ps[i].push.remote(keys, values) for i in range(opt.num_nodes)]

    def start(self):
        self.p1.start()
        self.p1.join(10)

    def end(self):
        self.p1.terminate()


@ray.remote(num_cpus=2, num_gpus=1, max_calls=1)
def worker_train(ps, node_buffer, opt, learner_index):
    agent = Learner(opt, job="learner")
    keys = agent.get_weights()[0]
    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)

    cache = Cache(node_buffer)

    cache.start()

    cnt = 1
    while True:
        batch = cache.q1.get()
        agent.train(batch, cnt)

        if cnt % opt.push_freq == 0:
            cache.q2.put(agent.get_weights())
        cnt += 1


@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):
    agent = Actor(opt, job="worker")
    keys = agent.get_weights()[0]
    np.random.seed()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT)
    from trading_env import TradingEnv, FrameStack
    # ------ env set up ------
    # env = gym.make(opt.env_name)
    env = TradingEnv(action_scheme_id=3, obs_dim=38)

    while True:

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)

        # for a_l_ratio control
        np.random.seed()
        rand_buff = np.random.choice(opt.num_buffers, 1)[0]
        last_learner_steps, last_actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

        while True:

            # don't need to random sample action if load weights from local.
            if last_actor_steps * opt.num_buffers > opt.start_steps or opt.recover:
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
            # d = False if ep_len*opt.action_repeat >= opt.max_ep_len else d

            np.random.seed()
            rand_buff = np.random.choice(opt.num_buffers, 1)[0]
            replay_buffer[rand_buff].store.remote(o, a, r, o2, d, worker_index)

            o = o2

            # End of episode. Training (ep_len times).
            # if d or (ep_len * opt.action_repeat >= opt.max_ep_len):
            if d:
                break


@ray.remote
def worker_test(ps, node_buffer, opt):

    agent = Actor(opt, job="test")
    keys = agent.get_weights()[0]

    # test_env = gym.make(opt.env_name)
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT)
    from trading_env import TradingEnv, FrameStack
    test_env = TradingEnv(action_scheme_id=3, obs_dim=38)

    init_time = time.time()
    save_times = 0
    checkpoint_times = 0

    while True:
        # weights_all for save it to local
        weights_all = ray.get(ps.get_weights.remote())
        weights = [weights_all[key] for key in keys]
        agent.set_weights(keys, weights)

        start_actor_step, start_learner_step, _ = get_al_status(node_buffer)
        start_time = time.time()

        ave_test_reward, ave_score = agent.test(test_env, 10)

        last_actor_step, last_learner_step, _ = get_al_status(node_buffer)
        actor_step = np.sum(last_actor_step) - np.sum(start_actor_step)
        learner_step = np.sum(last_learner_step) - np.sum(start_learner_step)
        alratio = actor_step / (learner_step + 1)
        update_frequency = int(learner_step / (time.time() - start_time))

        print("---------------------------------------------------")
        print("average test reward:", ave_test_reward)
        print("average test score:", ave_score)
        print("actor_steps:", last_actor_step, "learner_step:", last_learner_step)
        print("frame freq:", np.round((last_actor_step - start_actor_step) / (time.time() - start_time)))
        print("actor leaner ratio: %.2f" % alratio)
        print("learner freq:", update_frequency)
        print("Ray total resources:", ray.cluster_resources())
        print("available resources:", ray.available_resources())
        print("---------------------------------------------------")
        if learner_step < 1000:
            alratio = 0
        agent.write_tb(ave_test_reward, ave_score, alratio, update_frequency, last_learner_step)

        total_time = time.time() - init_time

        if last_learner_step // opt.save_interval > save_times:
            with open(opt.save_dir + "/" + str(last_learner_step / 1e6) + "M_" + str(ave_test_reward) + "_weights.pickle", "wb") as pickle_out:
                pickle.dump(weights_all, pickle_out)
                print("****** Weights saved by time! ******")
            save_times = last_learner_step // opt.save_interval

        # save everything every checkpoint_freq s
        if total_time // opt.checkpoint_freq > checkpoint_times:
            print("save everything!")
            save_start_time = time.time()

            ps_save_op = [node_ps[i].save_weights.remote() for i in range(opt.num_nodes)]
            buffer_save_op = [node_buffer[node_index][i].save.remote() for i in range(opt.num_buffers) for node_index in range(opt.num_nodes)]
            ray.wait(buffer_save_op + ps_save_op, num_returns=opt.num_nodes*opt.num_buffers + 1)

            print("total time for saving :", time.time() - save_start_time)
            checkpoint_times = total_time // opt.checkpoint_freq


def get_al_status(node_buffer):

    buffer_learner_step = []
    buffer_actor_step = []
    buffer_cur_size = []

    for node_index in range(opt.num_nodes):
        for i in range(opt.num_buffers):
            learner_step, actor_step, cur_size = ray.get(node_buffer[node_index][i].get_counts.remote())
            buffer_learner_step.append(learner_step)
            buffer_actor_step.append(actor_step)
            buffer_cur_size.append(cur_size)

    return np.array(buffer_actor_step), np.array(buffer_learner_step), np.array(buffer_cur_size)


if __name__ == '__main__':

    # ray.init()
    ray.init(resources={"node0": 256})

    # env = gym.make(FLAGS.env_name)
    env = TradingEnv(action_scheme_id=3, obs_dim=38)

    # ------ HyperParameters ------
    opt = HyperParameters(env, FLAGS.env_name, FLAGS.exp_name, FLAGS.num_nodes, FLAGS.num_workers, FLAGS.a_l_ratio, FLAGS.weights_file)

    if FLAGS.recover:
        opt.recover = True
    # ------ end ------

    node_ps = []
    node_buffer = []

    for node_index in range(FLAGS.num_nodes):

        # ------ Parameter Server (ray actor) ------
        # create model to get weights and create a parameter server
        node_ps.append(ParameterServer._remote(args=[opt, FLAGS.weights_file, FLAGS.checkpoint_path, node_index], resources={"node"+str(node_index): 1}))
        print(f"Node{node_index} Parameter Server all set.")
        # ------ Parameter Server end ------

        # ------ Experience buffer (ray actor) ------
        node_buffer.append([ReplayBuffer._remote(args=[opt, i+node_index*opt.num_buffers], resources={"node"+str(node_index): 1}) for i in range(opt.num_buffers)])

        if FLAGS.recover:
            buffer_load_op = [node_buffer[node_index][i].load.remote(FLAGS.checkpoint_path) for i in range(opt.num_buffers)]
            ray.wait(buffer_load_op, num_returns=opt.num_buffers)
        print(f"Node{node_index} Experience buffer all set.")
        # ------ Experience buffer end ------

        # ------ roll out worker (ray task) ------
        for i in range(FLAGS.num_workers):
            worker_rollout._remote(args=[node_ps[node_index], node_buffer[node_index], opt, i+node_index*FLAGS.num_workers], resources={"node"+str(node_index): 1})
            time.sleep(0.19)

        print(f"Node{node_index} roll out worker all up.")
        # ------ roll out worker end ------

    print(f"num of ps up: {len(node_ps)}, num of buffer up: {len(node_buffer)*len(node_buffer[0])}")

    print("Ray total resources:", ray.cluster_resources())
    print("available resources:", ray.available_resources())

    # --- save nodes info ---
    nodes_info = {
        "node_buffer": np.array(node_buffer),
        "num_nodes": opt.num_nodes,
        "num_buffers": opt.num_buffers
    }
    f_name = './nodes_info.pickle'
    with open(f_name, "wb") as pickle_out:
        pickle.dump(nodes_info, pickle_out)
        print("****** save nodes_info ******")
    # --- end ---

    # control learner start time
    if not opt.recover:

        start_time = time.time()

        total_cur_size = 0
        while total_cur_size < opt.start_steps:

            buffer_actor_step, buffer_learner_step, buffer_cur_size = get_al_status(node_buffer)
            total_cur_size = np.sum(buffer_cur_size)

            print("---------------------------------------------------")
            print("learner_step:", buffer_learner_step, "actor_steps:", buffer_actor_step)
            print("frame freq:", np.round(buffer_actor_step/(time.time()-start_time)))
            print("total frame freq:", int(np.sum(buffer_actor_step)/(time.time()-start_time)))
            print('start steps before learning:', total_cur_size, '/', opt.start_steps)
            print("Ray total resources:", ray.cluster_resources())
            print("available resources:", ray.available_resources())
            print("---------------------------------------------------")
            time.sleep(10)
    else:
        time.sleep(0.0)

    # ------ learner ------
    task_train = worker_train._remote(args=[node_ps[0], node_buffer, opt, 0], resources={"node0": 1})
    # ------ learner end ------

    task_test = worker_test.remote(node_ps[0], node_buffer, opt)
    ray.wait([task_test])
