import numpy as np
import tensorflow as tf
import time
import ray

from hyperparams_gfootball import HyperParameters, FootballWrapper
from actor_learner import Actor, Learner

import os
import pickle
import multiprocessing
import copy

from collections import deque

import inspect
import json
from ray.rllib.utils.compression import pack, unpack

import gfootball.env as football_env

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# "1_vs_1_easy" '11_vs_11_competition' '11_vs_11_stochastic'
flags.DEFINE_string("env_name", "11_vs_11_competition", "game env")
flags.DEFINE_string("exp_name", "Exp2", "experiments name")
flags.DEFINE_integer("num_workers", 6, "number of workers")
flags.DEFINE_string("weights_file", "", "empty means False. "
                                        "[Maxret_weights.pickle] means restore weights from this pickle file.")
flags.DEFINE_string("weights_folder_path", "", "empty means False. ")
flags.DEFINE_string("ext_weights_folder_path", "", "empty means False. ")
flags.DEFINE_float("a_l_ratio", 200, "actor_steps / learner_steps")
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
        if opt.obs_shape != (115,):
            self.buffer_o = np.array([['0' * 2000] * (opt.buffer_store_len + 1)] * opt.buffer_size, dtype=np.str)
        else:
            self.buffer_o = np.zeros((opt.buffer_size, opt.buffer_store_len+1) + opt.obs_shape, dtype=np.float32)
        self.buffer_a = np.zeros((opt.buffer_size, opt.buffer_store_len) + opt.act_shape, dtype=np.float32)
        self.buffer_r = np.zeros((opt.buffer_size, opt.buffer_store_len), dtype=np.float32)
        self.buffer_d = np.zeros((opt.buffer_size, opt.buffer_store_len), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, opt.buffer_size
        self.actor_steps, self.learner_steps = 0, 0

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
        self.actor_steps += self.opt.buffer_store_len * self.opt.num_buffers
        # self.actor_steps += self.buffer_store_len * self.action_repeat * self.opt.num_buffers
        # self.actor_steps += opt.Ln * opt.action_repeat

    def sample_batch(self):
        idxs = np.random.randint(0, self.size, size=self.opt.batch_size)
        idxs2 = np.random.randint(0, self.opt.buffer_store_len-self.opt.Ln, size=1)[0]

        # print(self.opt.buffer_store_len)
        self.learner_steps += 1 * self.opt.num_buffers
        # print(idxs2)
        # print(self.buffer_o.shape)
        # obs = self.buffer_o[idxs][:, idxs2:idxs2 + self.opt.Ln]
        # print(obs.shape)

        # buffer_o shape: (buffer size, max_ep_len, obs)
        # speed up slice using fancy indexing and broadcasting
        return dict(obs=self.buffer_o[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln + 1)],
                    acts=self.buffer_a[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)],
                    rews=self.buffer_r[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)],
                    done=self.buffer_d[idxs[:, None], np.arange(idxs2, idxs2 + self.opt.Ln)], )

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
        self.weights_pool = []
        self.ext_weights_pool = []

        if not checkpoint_path:
            checkpoint_path = opt.save_dir + "/checkpoint"

        if opt.recover:
            with open(checkpoint_path + "/checkpoint_weights.pickle", "rb") as pickle_in:
                self.weights = pickle.load(pickle_in)
                print("****** weights restored! ******")
            with open(checkpoint_path + "/checkpoint_weights_pool.pickle", "rb") as pickle_in:
                self.weights_pool = pickle.load(pickle_in)
                print("****** weights pool restored! ******")

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
            self.weights_pool = [copy.deepcopy(self.weights)]

    def push(self, keys, values):
        values = [value.copy() for value in values]
        for key, value in zip(keys, values):
            self.weights[key] = value

    def pool_push(self, weights=None):
        if not weights:
            weights = copy.deepcopy(self.weights)
        if len(self.weights_pool) < self.opt.num_in_pool:
            self.weights_pool.append(weights)
        else:
            self.weights_pool[np.random.choice(self.opt.num_in_pool, 1)[0]] = weights
        # pool_pop_ratio
        np.random.seed()
        if np.random.random() < opt.pool_pop_ratio:
            self.weights_pool.pop(0)

    def pool_pull(self, keys):
        # if np.random.random() < 0.2:
        #     worker_weights = self.weights
        # else:
        worker_weights = self.weights_pool[np.random.choice(len(self.weights_pool), 1)[0]]
        return [worker_weights[key] for key in keys]

    def ext_pool_pull(self, keys):
        worker_weights = self.ext_weights_pool[np.random.choice(len(self.ext_weights_pool), 1)[0]]
        return [worker_weights[key] for key in keys]

    def load_ext_pool(self, ext_weights_folder_path):
        import os
        weights_names = os.listdir(ext_weights_folder_path)
        for weights_name in weights_names:
            with open(ext_weights_folder_path + "/" + weights_name, "rb") as pickle_in:
                weights = pickle.load(pickle_in)
                self.ext_weights_pool.append(weights)
                print(weights_name, "in")
        print("load ext weights in pool all done")

    def pull(self, keys):
        return [self.weights[key] for key in keys]

    def get_weights(self):
        return copy.deepcopy(self.weights)

    # save weights to disk
    def save_weights(self):
        with open(opt.save_dir + "/checkpoint/" + "checkpoint_weights.pickle", "wb") as pickle_out:
            pickle.dump(self.weights, pickle_out)
        with open(opt.save_dir + "/checkpoint/" + "checkpoint_weights_pool.pickle", "wb") as pickle_out:
            pickle.dump(self.weights_pool, pickle_out)
        print("****** checkpoint weights and weights_pool saved! ******")


def load_weights_in_pool(ps, weights_folder_path):
    import os
    weights_names = os.listdir(weights_folder_path)
    for weights_name in weights_names:
        with open(weights_folder_path+"/"+weights_name, "rb") as pickle_in:
            weights = pickle.load(pickle_in)
            ps.pool_push.remote(weights=weights)
            print(weights_name, "in")
    print("load weights in pool all done")


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
        batch = cache.q1.get()
        if opt.model == "cnn":
            batch['obs'] = np.array([[unpack(o) for o in lno] for lno in batch['obs']])
        agent.train(batch, cnt)
        # TODO cnt % 300 == 0 before
        if cnt % 100 == 0:
            cache.q2.put(agent.get_weights())
        if cnt % opt.pool_push_freq == 0:
            ps.pool_push.remote()
        cnt += 1


@ray.remote
def worker_rollout_self_play(ps, replay_buffer, opt, worker_index):
    our_agent = Actor(opt, job="worker")
    opp_agent = Actor(opt, job="worker")
    keys = our_agent.get_weights()[0]

    random_steps = 0

    # ------ env set up ------

    env = football_env.create_environment(env_name=opt.rollout_env_name,
                                          number_of_left_players_agent_controls=1,
                                          number_of_right_players_agent_controls=1,
                                          stacked=opt.stacked, representation=opt.representation, render=False)

    env = FootballWrapper(env, opt.action_repeat, opt.reward_scale)
    # ------ env set up end ------

    while True:

        # sides = {'left':0, 'right':1}
        np.random.seed()
        if np.random.random() < opt.left_side_ratio:
            our_side = 0
            opp_side = 1
        else:
            our_side = 1
            opp_side = 0

        ################################## deques

        our_o_queue = []
        our_a_r_d_queue = []

        # left_o_queue = deque([], maxlen=opt.Ln + 1)
        # left_a_r_d_queue = deque([], maxlen=opt.Ln)

        # right_o_queue = deque([], maxlen=opt.Ln + 1)
        # right_a_r_d_queue = deque([], maxlen=opt.Ln)

        ################################## deques

        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        ################################## deques reset

        our_o = o[our_side]
        # left_o = o[0]
        # right_o = o[1]

        if opt.model == "cnn":
            our_compressed_o = pack(our_o)
            our_o_queue.append((our_compressed_o,))

            # right_compressed_o = pack(right_o)
            # right_o_queue.append((right_compressed_o,))
        else:
            our_o_queue.append((our_o,))
            # right_o_queue.append((right_o,))

        ################################## deques reset

        weights = ray.get(ps.pull.remote(keys))
        is_self_play = "self_play"
        our_agent.set_weights(keys, weights)
        np.random.seed()
        if np.random.random() < opt.self_pool_probability:
            weights = ray.get(ps.pool_pull.remote(keys))
            is_self_play = "self pool"
        elif np.random.random() < opt.ext_pool_probability:
            weights = ray.get(ps.ext_pool_pull.remote(keys))
            is_self_play = "ext pool"

        opp_agent.set_weights(keys, weights)

        # for a_l_ratio control
        np.random.seed()
        rand_buff = np.random.choice(opt.num_buffers, 1)[0]
        last_learner_steps, last_actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

        while True:

            # don't need to random sample action if load weights from local.
            if random_steps > opt.start_steps or opt.weights_file or opt.recover:
                our_opp_actions = [our_agent.get_action(o[our_side], False), opp_agent.get_action(o[opp_side], False)]
                a = [our_opp_actions[our_side], our_opp_actions[opp_side]]
            else:
                a = env.action_space.sample()
                random_steps += 1

            our_action = a[our_side]
            # left_action = a[0]
            # right_action = a[1]

            # Step the env
            o2, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1

            o = o2

            #################################### deques store

            our_o2 = o2[our_side]
            # left_o2 = o2[0]
            # right_o2 = o2[1]
            # #BUG a changed before here
            our_a_r_d_queue.append((our_action, r[our_side], d,))
            # left_a_r_d_queue.append((left_action, r[0], d,))
            # right_a_r_d_queue.append((right_action, r[1], d,))

            if opt.model == "cnn":
                our_compressed_o2 = pack(our_o2)
                our_o_queue.append((our_compressed_o2,))
                # left_compressed_o2 = pack(left_o2)
                # left_o_queue.append((left_compressed_o2,))
                # right_compressed_o2 = pack(right_o2)
                # right_o_queue.append((right_compressed_o2,))
            else:
                our_o_queue.append((our_o2,))
                # left_o_queue.append((left_o2,))
                # right_o_queue.append((right_o2,))

            # End of episode. Training (ep_len times).
            if d or (ep_len * opt.action_repeat >= opt.max_ep_len):

                replay_buffer[np.random.choice(opt.num_buffers, 1)[0]].store.remote(our_o_queue, our_a_r_d_queue,
                                                                                    worker_index)
                learner_steps, actor_steps, _ = ray.get(replay_buffer[rand_buff].get_counts.remote())
                print('rollout_ep_len:', ep_len * opt.action_repeat, 'our_side:', our_side, 'is_self_play:', is_self_play, 'rollout_ep_ret:', ep_ret[our_side])

                # for a_l_ratio control
                learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())
                while (actor_steps - last_actor_steps) / (learner_steps - last_learner_steps + 1) > opt.a_l_ratio and last_learner_steps > 0:
                    time.sleep(1)
                    learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

                break


@ray.remote
def worker_rollout_bot(ps, replay_buffer, opt, worker_index):
    rollout_di_env_name = "11_vs_11_stochastic"

    agent = Actor(opt, job="worker")
    keys = agent.get_weights()[0]
    np.random.seed()
    rand_buff1 = np.random.choice(opt.num_buffers, 1)[0]

    if opt.recover:
        learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff1].get_counts.remote())
        mu = min(learner_steps / opt.mu_speed, 1.0)
        sigma = 0.15
    else:
        mu, sigma = 0, 0.2

    random_steps = 0

    while True:
        # ------ env set up ------

        while True:
            np.random.seed()
            s = np.random.normal(mu, sigma, 1)
            if 0 < s[0] < 1:
                using_difficulty = int(s[0] // 0.05 + 1)
                break

        env = football_env.create_environment(env_name=rollout_di_env_name + '_' + str(using_difficulty),
                                              stacked=opt.stacked, representation=opt.representation, render=False)

        env = FootballWrapper(env, opt.action_repeat, opt.reward_scale)
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

                learner_steps, actor_steps, _ = ray.get(replay_buffer[rand_buff].get_counts.remote())
                print('rollout_ep_len:', ep_len * opt.action_repeat, 'mu:', mu, 'using_difficulty:', using_difficulty,
                      'rollout_ep_ret:', ep_ret)

                if mu < 1:
                    mu = learner_steps / opt.mu_speed
                else:
                    mu = 1

                # for a_l_ratio control
                learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())
                while (actor_steps - last_actor_steps) / (learner_steps - last_learner_steps + 1) > opt.a_l_ratio and last_learner_steps > 0:
                    time.sleep(1)
                    learner_steps, actor_steps, _size = ray.get(replay_buffer[rand_buff].get_counts.remote())

                break


@ray.remote
def worker_test(ps, replay_buffer, opt):
    agent = Actor(opt, job="main")

    test_env = football_env.create_environment(env_name="11_vs_11_easy_stochastic",
                                               stacked=opt.stacked, representation=opt.representation,
                                               render=False)

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
    All_Parameters["wrapper"] = inspect.getsource(FootballWrapper)
    import importlib

    scenario = importlib.import_module('gfootball.scenarios.{}'.format(opt.rollout_env_name))
    All_Parameters["rollout_env_class"] = inspect.getsource(scenario.build_scenario)
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

    if FLAGS.weights_folder_path:
        load_weights_in_pool(ps, FLAGS.weights_folder_path)

    if FLAGS.ext_weights_folder_path:
        ps.load_ext_pool.remote(FLAGS.ext_weights_folder_path)

    # Experience buffer
    # Methods called on different actors can execute in parallel,
    # and methods called on the same actor are executed serially in the order that they are called.
    # we need more buffer for more workers to keep high store speed.
    replay_buffer = [ReplayBuffer.remote(opt, i) for i in range(opt.num_buffers)]

    if FLAGS.recover:
        buffer_load_op = [replay_buffer[i].load.remote(FLAGS.checkpoint_path) for i in range(opt.num_buffers)]
        ray.wait(buffer_load_op, num_returns=opt.num_buffers)

    # Start some training tasks.
    num_bot_worker = int(opt.bot_worker_ratio * FLAGS.num_workers)
    for i in range(FLAGS.num_workers-num_bot_worker):
        worker_rollout_self_play.remote(ps, replay_buffer, opt, i)
        time.sleep(3)
    for i in range(num_bot_worker):
        worker_rollout_bot.remote(ps, replay_buffer, opt, i)
        time.sleep(3)
    # task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

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
