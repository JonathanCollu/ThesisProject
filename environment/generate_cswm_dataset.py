# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Template for running an agent on Spriteworld tasks.

This script runs an agent on a Spriteworld task. The agent takes random actions
and does not learn, so this serves only as an example of how to run an agent in
the environment, logging task success and mean rewards.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from absl import logging
import numpy as np
from six.moves import range

from spriteworld import environment
from spriteworld import renderers

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 1, 'Number of training episodes.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')

# tasks
flags.DEFINE_string('gfa',
                    'spriteworld.configs.examples.goal_finding_agent',
                    'Module name of task config to use.')

flags.DEFINE_string('gfi',
                    'spriteworld.configs.examples.goal_finding_interactive',
                    'Module name of task config to use.')

flags.DEFINE_string('clustering',
                    'spriteworld.configs.examples.clustering_interactive',
                    'Module name of task config to use.')

flags.DEFINE_string('sorting',
                    'spriteworld.configs.examples.sorting_interactive',
                    'Module name of task config to use.')


class RandomAgent(object):
  """Agent that takes random actions."""

  def __init__(self, env):
    """Construct random agent."""
    self._env = env

  def step(self, timestep):
    # observation is a dictionary with renderer outputs to be used for training
    observation = timestep.observation
    del observation
    del timestep
    action = self._env.action_space.sample()[1]
    return action


def save_list_dict_h5py(array_dict, fname):
  """Save list of dictionaries containing numpy arrays to h5py file."""

  # Ensure directory exists
  import h5py, os
  directory = os.path.dirname(fname)
  if not os.path.exists(directory):
      os.makedirs(directory)
  
  with h5py.File(fname, 'w') as hf:
    for i in range(len(array_dict)):
          grp = hf.create_group(str(i))
          for key in array_dict[i].keys():
              grp.create_dataset(key, data=array_dict[i][key])

def main(argv):
  del argv
  gfa = importlib.import_module(FLAGS.gfa)
  gfa = gfa.get_config(FLAGS.mode)
  gfa['renderers']['success'] = renderers.Success()  # Used for logging

  gfi = importlib.import_module(FLAGS.gfi)
  gfi = gfi.get_config(FLAGS.mode)
  gfi['renderers']['success'] = renderers.Success()  # Used for logging

  clustering = importlib.import_module(FLAGS.clustering)
  clustering = clustering.get_config(FLAGS.mode)
  clustering['renderers']['success'] = renderers.Success()  # Used for logging

  sorting = importlib.import_module(FLAGS.sorting)
  sorting = sorting.get_config(FLAGS.mode)
  sorting['renderers']['success'] = renderers.Success()  # Used for logging

  # Loop over episodes, logging success and mean reward per episode
  obs = []
  for episode in range(FLAGS.num_episodes):
    if episode % 4 == 0: env = environment.Environment(**gfa)
    if episode % 4 == 1: env = environment.Environment(**gfi)
    if episode % 4 == 2: env = environment.Environment(**clustering)
    if episode % 4 == 3: env = environment.Environment(**sorting)
    agent = RandomAgent(env)
    timestep = env.reset()
    rewards = []
    obs.append({'obs': [], 'action': [], 'next_obs': []})
    while not timestep.last():
      obs[episode]['obs'].append(timestep.observation["image"]/255)
      action = agent.step(timestep)
      print(action)
      obs[episode]['action'].append(action)
      timestep = env.step(action)
      obs[episode]['next_obs'].append(timestep.observation["image"]/255)
      rewards.append(timestep.reward)
    logging.info('Episode %d: Success = %r, Reward = %s.', episode,
                 timestep.observation['success'], np.nanmean(rewards))
  save_list_dict_h5py(obs, 'data/spriteworld.h5')

if __name__ == '__main__':
  app.run(main)
