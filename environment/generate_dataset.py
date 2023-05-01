
import importlib
from absl import app
from absl import flags

from spriteworld import environment
from spriteworld import renderers

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100, 'Number of training episodes.')
flags.DEFINE_string('config',
                    'spriteworld.configs.examples.goal_finding_embodied',
                    'Module name of task config to use.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')

def main(argv):
  del argv
  config = importlib.import_module(FLAGS.config)
  config = config.get_config(FLAGS.mode)
  config['renderers']['success'] = renderers.Success()  # Used for logging
  env = environment.Environment(**config)
  env.get_image_dataset()

if __name__ == '__main__':
  app.run(main)