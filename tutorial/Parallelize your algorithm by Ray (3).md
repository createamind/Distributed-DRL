# 使用Ray并行化你的强化学习算法（三）

## SAC并行版本实现

这一章，我们将上节分好的各部分代码放入并行框架中。

我们的并行框架结构图（内容仅涉及到白色线条部分）：

![ddrlframework](.\Pictures\ddrlframework.jpg)

下面是用ray实现的框架。

```python
@ray.remote
class ReplayBuffer:
	...
    # replay buffer
    
@ray.remote
class ParameterServer(object):
	...
    # keep the newest network weights here
    # could pull and push the weights
    # also could save the weights to local
    
@ray.remote
def worker_rollout(ps, replay_buffer, opt, worker_index):
    ...
    # bulid a rollout network
    # pull weights from ps
    # for loop:
    #	interactive with environment
    #	store experience to replay buffer
    #	if end of episode:
    #		pull weights from ps

@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, opt, learner_index):
    ...
    # build a learner network
    # pull weights from ps
  	# for loop:
    #	get sample batch from replaybuffer
    #	update network and push new weights to ps
    
@ray.remote
def worker_test(ps, replay_buffer, opt, worker_index=0):
    ...
    # bulid a test network usually same as rollout
    # while:
    #	pull weights from ps
    #	do test
    #	might save model here
    
if __name__ == '__main__':

    ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)

    # create the parameter server
    ps = ParameterServer.remote([], [], is_restore=True)

    # create replay buffer
    replay_buffer = ReplayBuffer.remote(obs_dim=opt.obs_dim, act_dim=opt.act_dim, size=opt.replay_size)

    # Start some rollout tasks
    task_rollout = [worker_rollout.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_workers)]

    time.sleep(5)
	
	# start training tasks
    task_train = [worker_train.remote(ps, replay_buffer, opt, i) for i in range(FLAGS.num_learners)]

    # start testing
    task_test = worker_test.remote(ps, replay_buffer, opt)

    # wait util task test end
    # Keep the main process running. Otherwise everything will shut down when main process finished.
    ray.wait([task_test, ])
```

---

0. model

我们先看算法的核心部分：model，包含了TensorFlow建图，计算loss，训练和测试。

新建一个的文件，将之前model部分，训练部分和测试部分的代码都放入Model类中去。之后我们建立一个实例后，就可以调用方法生成动作，训练更新参数，测试评估参数。

```python
class Model(object):

    def __init__(self, args):
        # model part code
    def get_action(self, o, deterministic=False):
        # get_action method
    def train(self, replay_buffer, args):
        # train part code
    def test_agent(self, test_env, args, n=10):
        # test method copy
        
```

---

将代码放入对应位置。

```python
import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.sac import core
from spinup.algos.sac.core import get_vars
from spinup.utils.logx import EpochLogger
from core import mlp_actor_critic as actor_critic
import ray.experimental.tf_utils


class Model(object):

    def __init__(self, args):

        # Inputs to computation graph


    def get_action(self, o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={self.x_ph: o.reshape(1, -1)})[0]

    def train(self, replay_buffer, args):

        for j in range(args.ep_len):
            batch = replay_buffer.sample_batch(args.batch_size)
            feed_dict = {self.x_ph: batch['obs1'],
                         self.x2_ph: batch['obs2'],
                         self.a_ph: batch['acts'],
                         self.r_ph: batch['rews'],
                         self.d_ph: batch['done'],
                         }
            outs = sess.run(self.step_ops, feed_dict)
            # logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
            #              LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
            #              VVals=outs[6], LogPi=outs[7])

    def test_agent(self, test_env, args, n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not (d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            print(ep_len, ep_ret)
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

```

---

之外，我们还需要额外添加几个有用的方法。learner不断更新权重，需要把最新的权重导出到ps server上去。rollout需要不断从ps上下载最新权重并更换为自己的权重。

ray中已经有写好的类。方便我们导入和导出权重。

```python
    def __init__(self, args):
        
        ...
        
		self.variables = ray.experimental.tf_utils.TensorFlowVariables(self.value_loss, self.sess)
```

目标函数的权重在导入权重以后做初始化才有意义，所以把它放在更新权重方法里。

```python
    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))
        self.sess.run(self.target_init)

    def get_weights(self):
        weights = self.variables.get_weights()
        keys = [key for key in list(weights.keys()) if "main" in key]
        values = [weights[key] for key in keys]
        return keys, values
```

---

1. Replay Buffer，只要在上面加上ray的修饰器就行了。

```python
@ray.remote
class ReplayBuffer:
	...
    # replay buffer
```

---

2. Parameter Server

参数保存在字典里面。Parameter Server的主要功能就是给worker返回最新的权重，接收learner传来的最新的权重。

```python
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
```

---

3. rollout

rollout (worker) 与环境交互，产生数据并存入Replay Buffer。每个episode结束会从Parameter Server得到最新权重来更新自己。

```python
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
```

---

4. train

我们使用一个GPU进行训练。所有在ray修饰器里我们设置资源请求量。

当使用GPU执行任务时，任务会在GPU上分配内存，而且有可能在执行结束后不释放。在设置中写入`max_calls=1`可以让任务运行结束后自动退出并释放GPU内存。

```python
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
```

---

5. test

```python
@ray.remote
def worker_test(ps, start_time):

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger = EpochLogger(**logger_kwargs)
    # print(locals())
    # logger.save_config(locals())

    agent = Model(args)
    keys = agent.get_weights()[0]

    weights = ray.get(ps.pull.remote(keys))
    agent.set_weights(keys, weights)
    test_env = gym.make(args.env)
    while True:
        ave_ret = agent.test_agent(test_env, args)
        # print("test Average Ret:", ave_ret, "time:", time.time()-start_time)
        logger.log_tabular('test Average Ret', ave_ret)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
        weights = ray.get(ps.pull.remote(keys))
        agent.set_weights(keys, weights)

```

---

主程序调用

```python
if __name__ == '__main__':
    
    ...
    
    ray.init()

    net = Model(args)
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values)

    replay_buffer = ReplayBuffer.remote(args.obs_dim, args.act_dim, args.replay_size)

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, args) for i in range(args.num_workers)]

    time.sleep(20)

    task_train = [worker_train.remote(ps, replay_buffer, args) for i in range(args.num_learners)]

    time.sleep(10)

    task_test = worker_test.remote(ps)
    ray.wait([task_test, ])
```

本节完。

本文展示的代码是实现分布式算法的最小改动版本，还有许多地方可以优化。

简单实验对比：

实验：LunarLanderContinuous-v2

未调参，sac和dsac参数相同，dsac的worker数量：1。GPU：GTX1060

![dsac1w-sac](.\Pictures\dsac1w-sac.png)

完整代码链接：<https://github.com/createamind/Distributed-DRL/tree/master/example>

参考资料：

https://ray.readthedocs.io/en/master/auto_examples/plot_parameter_server.html