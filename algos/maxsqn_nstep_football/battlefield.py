import tensorflow as tf
from hyperparams_gfootball import HyperParameters
from actor_learner import Actor
import pickle
import gfootball.env as football_env

# flags = tf.app.flags
# FLAGS = tf.app.flags.FLAGS
# "GoodWeights/343/Max_weights.pickle", ""

# left_team_weights = "GoodWeights/343/343E1.pickle"
# right_team_weights = "GoodWeights/343/1M_weights.pickle"

right_team_weights = "9.16Max_weights.pickle"
left_team_weights = "9.16Max_weights.pickle"
# 15 2.66 l50 -0.88
# "11_vs_11_competition", "11_vs_11_easy_stochastic"-1.74
env_name = "11_vs_11_competition"

opt = HyperParameters(env_name, '', 0, 0, '')

opt.hidden_size = (300, 400, 300)
left_agent = Actor(opt, job="test")
keys, _ = left_agent.get_weights()

with open(left_team_weights, "rb") as pickle_in:
    left_weights_all = pickle.load(pickle_in)
left_weights = [left_weights_all[key] for key in keys]
left_agent.set_weights(keys, left_weights)

opt.hidden_size = (300, 400, 300)
right_agent = Actor(opt, job="test")
keys, _ = right_agent.get_weights()
with open(right_team_weights, "rb") as pickle_in:
    right_weights_all = pickle.load(pickle_in)
right_weights = [right_weights_all[key] for key in keys]
right_agent.set_weights(keys, right_weights)

test_env = football_env.create_environment(env_name=opt.env_name, stacked=opt.stacked,
                                           number_of_left_players_agent_controls=1,
                                           number_of_right_players_agent_controls=1,
                                           representation=opt.representation, render=True)

n = 50

rew = []
for j in range(1, n + 1):
    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0

    while not d:
        left_action = left_agent.get_action(o[0], True)
        # left_action = test_env.action_space.sample()[0]

        # right_action = test_env.action_space.sample()[0]
        right_action = right_agent.get_action(o[1], True)
        action = [left_action, right_action]

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.03)
        ep_ret += r
        ep_len += 1

    rew.append(ep_ret[0])
    print("score: ", ep_ret)
    print("ave test reward:", sum(rew) / j, j)
print("ave test_reward:", sum(rew) / n)
