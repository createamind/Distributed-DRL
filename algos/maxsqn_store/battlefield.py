import tensorflow as tf
from hyperparams_gfootball import HyperParameters
from actor_learner import Actor
import pickle
import gfootball.env as football_env

# flags = tf.app.flags
# FLAGS = tf.app.flags.FLAGS



left_team_weights = "agents/103.573526M_6.7_weights.pickle"#102.624116M_6.96_weights.pickle"#103.085962M_6.98_weights.pickle"#102.100236M_6.76_weights.pickle"#102.760798M_7.3_weights.pickle"#94.54844M_6.18_weights.pickle"#98.614691M_6.9_weights.pickle"#99.487003M_7.32_weights.pickle"#98.49637M_6.78_weights.pickle"#97.238274M_6.6_weights.pickle"#95.367972M_7.7_weights.pickle"#96.853064M_7.12_weights.pickle"#94.470649M_7.52_weights.pickle"#95.509652M_7.22_weights.pickle"#93.552998M_7.2_weights.pickle"#93.171218M_4.48_weights.pickle"#92.399664M_7.84_weights.pickle"#89.954949M_7.14_weights.pickle"#93.949107M_5.56_weights.pickle"#93.750531M_5.1_weights.pickle"#93.153151M_4.2_weights.pickle"#95.152022M_4.92_weights.pickle"#94.54844M_6.18_weights.pickle"#91.578249M_4.74_weights.pickle"#88.816672M_4.46_weights.pickle"#89.594806M_4.34_weights.pickle"#90.580364M_5.74_weights.pickle"#91.578249M_4.74_weights.pickle"#92.355662M_6.02_weights.pickle"#93.549288M_5.82_weights.pickle"#94.54844M_6.18_weights.pickle"#95.152022M_4.92_weights.pickle"#96.319342M_4.48_weights.pickle"#97.279252M_5.0_weights.pickle"#98.242557M_4.0_weights.pickle"#99.783999M_4.62_weights.pickle"#104.227424M_4.76_weights.pickle"#106.361906M_4.92_weights.pickle"#106.755215M_5.18Max_weights.pickle"#106.948044M_4.06_weights.pickle"#103.614035M_2.8_weights.pickle"#103.634041M_4.8Max_weights.pickle"#102.912908M_4.24_weights.pickle"#103.061805M_4.02_weights.pickle"#102.483514M_3.74Max_weights.pickle"#102.55931M_3.54Max_weights.pickle"
#right_team_weights = "rightrandom0.015_buffersize2.5e6_ar3_rs150_COPY10M_2/87.662722M_6.38_weights.pickle"
right_team_weights = "agents/94.54844M_6.18_weights.pickle"#103.061805M_4.02_weights.pickle"#101.507322M_1.72_weights.pickle"#101.369513M_3.74Max_weights.pickle"#101.150784M_3.14Max_weights.pickle"
#right_team_weights ="5.58782M_6.88_weights.pickle"
#right_team_weights ="46.663892M_6.34_weights.pickle"
print("left_team_weights:", left_team_weights)
print("right_team_weights:", right_team_weights)

# "9.16Max_weights.pickle"
# 11.644652M_7.12_weights.pickle
# 21.138408M_5.98_weights.pickle
# "M343Pool3*wPro0.5Keepratio20_Scale_180_OLD/6.029512M_5.82_weights.pickle"
# "M3432P1000_10k_Pro0.5_KeepratioE111_OLD_Scale180_alpha0.1_E1/3.031384M_4.42_weights.pickle"


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
                                           representation=opt.representation, render=False)

n = 1000

rew = []
for j in range(1, n + 1):

    test_env = football_env.create_environment(env_name=opt.env_name, stacked=opt.stacked, number_of_left_players_agent_controls=1, number_of_right_players_agent_controls=1,representation=opt.representation, render=False)

    o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
    retl,retr=0,0
    cnt_pause = 0
    while not d:
        left_action = left_agent.get_action(o[0], True)
        right_action = right_agent.get_action(o[1], True)
        
        # handling env stuck
        if cnt_pause > 0 and o[0][108]:
            print("env getting stuck.....", cnt_pause, 'steps')
            cnt_pause = 0
        if not o[0][108]:
            cnt_pause += 1
            if cnt_pause <= 50:
                left_action = left_agent.get_action(o[0], False)
                right_action = right_agent.get_action(o[1], False)
            else:
                left_action = test_env.action_space.sample()[0]
                right_action = test_env.action_space.sample()[0]

        action = [left_action, right_action]

        o, r, d, _ = test_env.step(action)
        # time.sleep(0.03)
        ep_ret += r
        ep_len += 1
        if r[0]>0:
            retl += 1
        if r[1]>0:
            retr += 1
    rew.append(ep_ret[0])

    if j%30 == 0:
        print("left_team_weights:", left_team_weights)
        print("right_team_weights:", right_team_weights)

    print("score: ",ep_ret[0],"[",retl,":",retr, "]")
    print("ave score reward:", sum(rew) / j, j)

print("left_team_weights:", left_team_weights)
print("right_team_weights:", right_team_weights)
print("ave score reward:", sum(rew) / n)
