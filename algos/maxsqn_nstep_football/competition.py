# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example football client.
It creates remote football game with given credentials and plays a few games.
"""

from absl import app
from absl import flags
import gfootball.env as football_env
from gfootball.env import football_action_set
import grpc

from hyperparams_gfootball import HyperParameters
from actor_learner import Actor
import pickle
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

FLAGS = flags.FLAGS
flags.DEFINE_string('username', 'createamind', 'Username to use')
flags.mark_flag_as_required('username')
flags.DEFINE_string('token', '4HESaBAE1Q_1573465228', 'Token to use.')
flags.DEFINE_integer('how_many', 10, 'How many games to play')
flags.DEFINE_bool('render', False, 'Whether to render a game.')
flags.DEFINE_string('track', '11vs11', 'Name of the competition track.')
flags.DEFINE_string('model_name', 'kangaroo-easy',
                    'A model identifier to be displayed on the leaderboard.')
flags.DEFINE_string('inference_model', '',
                    'A path to an inference model. Empty for random actions')

NUM_ACTIONS = len(football_action_set.action_set_dict['default'])


def main(unused_argv):
    env_name = "11_vs_11_competition"
    opt = HyperParameters(env_name, '', 0, 0, '')
    opt.hidden_size = (300, 400, 300)

    model = Actor(opt, job="test")
    keys, _ = model.get_weights()
    with open("9.16Max_weights.pickle", "rb") as pickle_in:
        weights_all = pickle.load(pickle_in)
    weights = [weights_all[key] for key in keys]

    model.set_weights(keys, weights)
    env = football_env.create_remote_environment(
        FLAGS.username, FLAGS.token, FLAGS.model_name, track=FLAGS.track,
        representation='simple115', stacked=False,
        include_rendering=FLAGS.render)
    for _ in range(FLAGS.how_many):
        ob = env.reset()
        cnt = 1
        done = False
        while not done:
            try:
                action = model.get_action(ob, True)
                logging.info('Before calling env.step')
                ob, rew, done, _ = env.step(action)
                logging.info('After calling env.step')
                print('Playing the game, step {}, action {}, rew {}, done {}'.format(
                    cnt, action, rew, done))
                cnt += 1
            except grpc.RpcError as e:
                print(e)
                break
        print('=' * 50)


if __name__ == '__main__':
    app.run(main)
