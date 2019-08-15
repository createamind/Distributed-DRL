# Distributed-DRL
Distributed Deep Reinforcement Learning

This framework inspired by general-purpose RL training system **Rapid** from OpenAI.

Rapid framework:
![rapid-architecture@2x--1-](https://openai.com/content/images/2018/06/rapid-architecture@2x--1-.png)
Our framework:
![ddrlframework](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0ff8d04c-31f0-425b-9021-b6138eadf94e/ddrlframework.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45KJBQMENX%2F20190815%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190815T130427Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEJT%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLXdlc3QtMiJHMEUCIFZKSCUX%2FGC4DQqWKejwdAYhfaRdUN0qPursVj1gNRk3AiEAibNCoCo7x%2BvaID8X6R1WFuXyvqOhVSq0jJfBznL4MxAq2gMILRAAGgwyNzQ1NjcxNDkzNzAiDDIBQJB9TpdQElV4Cyq3Azwvx%2FYgBNRvO11EF3mpnxieSJdoR9fXNdAOLRAcYPbkdg%2Bwn8kLSckr%2Fx7V9fIPFjV5P3BI1VczlGDrxUJGNKgqeo8THJ%2BGON86Hb592guBtLIsyDDxQOqy4Vuc2ckpPizAucREqVKsaMyCPzKQeQEXXFJ6IPK3qAiOBzYs1ByAhxO1I6VbvmoBiFXefZufkHjizv94paQo8u7NVz9c5ZV34%2Bjn%2FwBDtjZOSPydFeniAnuMsdCAYFrxJmxFjtZxQCDf%2BeZuts5hCK7jcYrLTOM%2FP9Gw86z2Uu5KbZRhgnnnX6VNteoznUw7AC3EH99fbhWgvAHPLLHIbEklptZYFSZf4EtKGQ%2FqE5rS9s3FhaROYBJnMcj8%2FeispwP603y9%2FZae0%2FTJ2WCN%2F5QFWoxai9Ev3WURGmw9Mr0dxzgNLX3%2BnrvFtxa2DAwTj48JOkvQ62ipaGpAcSooaXbYnhYB1drqiV%2BkZr7yEIHrAULspCV1V16ayORmoXrjgF9WmXn8M1irmZRjy4AQWcykx1ln6Msb96SHTVDBS1PxR02xi%2FG5GKlnWJbWHqbKhFfJ13u1lyblVJrm1dowrZPV6gU6tAFUvpZPMOHOoGL6A6oeeO8q1bNV%2F6pd5UqjXx4Cxrv20jMGic2c5Y4yycEcw3gaRFUgYotXVwJrmWK0k67y%2B6XWrljT4Z6FhQSR5jpfDu7f6drtb2Kviucdmtep3fzvu837kDElkWA8lBZGgZ%2Fl1x3sdJGfNEh2o7YGltNoQiHT0opVREOdyK3%2BW3PtRAtx7TQZVFqSqwHY9C%2F7F5JAQRr8bqd0jdkj9%2Bc53ZXiJZi9k1gRSME%3D&X-Amz-Signature=bae181fa6089f7564afc3348d0aebaa032f16a6440c1ec1ce8433fe63c36f1f8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22ddrlframework.jpg%22)

This framework divides the reinforcement learning process into five parts:

- Replay buffer (option)
- Parameter server
- train (learn)
- rollout
- test

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
    
@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, opt, learner_index):
    ...
    # build a learner network
    # pull weights from ps
  	# for loop:
    #	get sample batch from replaybuffer
    #	update network and push new weights to ps
    
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

    opt = HyperParameters(FLAGS.env_name, FLAGS.total_epochs, FLAGS.num_workers)

    # create the parameter server
    if FLAGS.is_restore == "True":
        ps = ParameterServer.remote([], [], is_restore=True)
    else:
        net = Learner(opt, job="main")
        all_keys, all_values = net.get_weights()
        ps = ParameterServer.remote(all_keys, all_values)

    # create replay buffer
    replay_buffer = ReplayBuffer.remote(obs_dim=opt.obs_dim, act_dim=opt.act_dim, size=opt.replay_size)

    # Start some rollout tasks.
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



### Result:

Env: LunarLanderContinuous-v2
GPU:GTX1060 x1

**SAC1 without distribution:** gets 200+ in 1200s
![sac1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/db6519ba-b95e-447b-93a7-a34f6142c744/sac1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45IU3V2MDH%2F20190814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190814T123307Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHgaCXVzLXdlc3QtMiJIMEYCIQCt1bPUIwSRH5YCCk0Zvfl6RWWT2FzL2OSUuQo9BbWjRQIhAKgXLNA9aJVESoC57CI1S%2BF0RKFqskl%2FIa4e1GYo044wKtoDCBAQABoMMjc0NTY3MTQ5MzcwIgxdXRPsEPCBphBdizcqtwPzeyc3VsMZsVoxgZP7mwrxESv5IXLp2hw4ycAjrwmSDquT3Dn1za%2FqRwNIIF8%2BQ1JFWbpvIEFVzpX2kCkWJltUNX0hjc4wtlna70f7Nf%2BreQV3qnylHApLuO88zyFAFrvKJdEzoxytkw1XfWzY0eSd1vevx5qPMKLMKmim9GlwNzgMoTkCTPoXSM%2BfZBaRwwimPQu2ObweUt5k6NasdUJSuPCKcdg7ptQnCdFj%2FFXoZZMotuknxcBsf59GBtl3QqcWXHnTljtiAZs5vm84GKjoM92CUs57SfruwmyZoAksWgxQYrifmcSNNS2bsjrFeI0xwOp61R7Aw9oUsCbVdZf%2F7qAh6urcc%2FdMWuRHmMRCDq3a2hB26L56%2FTxXOyDV7NgH6l%2BRw4R0sCKYO96n8OiXduxHHiJB7ZV88y8zSv5%2BPQYi7SRHyn1654qb363YweB%2FVNle4EcZr%2Bwvu3KkNdMQ0GDOphU4bIJlitLW9BI4K0bamHnKZw89DgWrI32uzzrim3MjkkkkIjvsAiLrpzIAF7gnXTduiIelcLuhyKJ36gskybR5waEIaff%2FFWe163vU%2FutyUO1%2BMKTrzuoFOrMBzCtLlnmwkUbV7WQMho2PuN5WXDarmJWlx9IgjshYvVuPdPTrBLpyLuOanjomZJaI1WEG%2FpgvHx4a2KeQtyzWVzT7KmLpkfY3dAajzzjSyRzof6h6frQwp0gFNdHgyogKMRZKRAdbw16p1QWm0jvQvDLwtw281d7zABLdmaudhjl3s%2Fu86p8Ksk4%2BTUr2CnjkYx37Rn1y6p8lJ4J2d3rRIjk%2BHljarwPfe7peXpSnr%2BofNb4%3D&X-Amz-Signature=728b3a7e5b901f289254868d78685adb7be0ee5f43b9cb6d916b38caaaefc929&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22sac1.png%22)
**Distributed SAC1:** gets 200+ in 360s
![dsac1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7303a5b5-beee-4452-aa8f-5a3743d3f4a2/dsac1.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAT73L2G45F2HMZQ5M%2F20190814%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20190814T123314Z&X-Amz-Expires=86400&X-Amz-Security-Token=AgoJb3JpZ2luX2VjEHgaCXVzLXdlc3QtMiJIMEYCIQCha3PdJsRMvAmRjRwYUzWOz6sO0C416ukfouGiDqiA1AIhAMzq2oRVEAF%2FxcdI%2BeMNPZ2mY2RZUj2BYpmGJGo7eTdOKtoDCBAQABoMMjc0NTY3MTQ5MzcwIgyjCZ1N%2FotP35Qg34IqtwNHqOxJpWV9M9Zsb7V%2FTItWKC0txTRw0%2F8OX4TCtcA%2F34IBzLeGcF8N%2BD9ZJ2Ie%2BcPo2HsevHcgiiQE2%2BxsAyEC%2BjGQURxwLyJVyywqkQMRRlVXjH8ykN9lT3uhFzIon4uChXsKJAudZfWkkDH8vJfNPFrlVLCGSt0eFOwRPRagI1gzGHurIXskaRJloL56JoCQmCgj%2F7Z3qJVj6MW7co4RHDWIL5OFxZwcKykG%2FRwR6eW7Tg4dk9wg8iFY53VCOaeqvszsXMTx5l85s%2BtYu7u%2BkrtN9IE8kERMF8XxauGqGsxhQr9m02NdlspSIfvNrbQjchMYMe0TTwrqeMDF6WmgRMED4%2FqS0NXSQQ0EoSh8VpymDE0HvBVL9hfR1%2FREh1hUsvf0o4wLV7UQ5RlMnIzl5kFM3ut0WWqiDswiwiDkxeDR3DLGJiRnlJzdna34o25zxsHXgKZKj6oEHiF10sWtSrxDtOGCFYtZVfCarEVvUJiELkb1z0BC8pw51ElRqtn0yAFk66oIyHPJEbCBbIFJCkgfn%2BnDxry8sdFcWs%2FSbvljBGG2hxJrE4JrCZyit0ZpbpKE2uY9MNjrzuoFOrMB5MLBSGQVRzBvRiXaF3q4D89oqmudyFTuslI9AdMcopSrIGZtAxdO2h%2B8XZGKMUGy9infGSU%2FIy8PzW9FowFbFR65LEZz7Ppsg5PHiMKVtgnAen2T8k%2Bh1fGk8Bq%2B6QaaPSo8qAPSgpjq4zanpEClDPgEvBwNY595EST7bbyEIiYW0XFObXS113v%2BT3TgbPlwtMwJXtvqSleCAXyu7EXE2QXffIpXTzxV0SohTO06VFM4xdk%3D&X-Amz-Signature=315a6bd8a1bfc265208eae88a562813c956d913d7b805b92127c1591f0c47b9a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22dsac1.png%22)
