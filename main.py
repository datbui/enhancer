import logging
import logging.config
import os
import pprint
from logging.handlers import RotatingFileHandler

import tensorflow as tf
import yaml

from config import FLAGS
from model import get_estimator, get_input_fn
from scripts.download import download_dataset
from utils import get_tfrecord_files, save_config

pp = pprint.PrettyPrinter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_logging(default_path='properties/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logger = logging.getLogger()
        logger.setLevel(default_level)
        # create file handler which logs even debug messages
        fh = RotatingFileHandler(os.path.join(FLAGS.log_dir, 'tensorflow_default.log'), maxBytes=10 * 1024 * 1024)
        fh.setLevel(default_level)
        formatter = logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def run_training(config, session):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config)

    filenames = get_tfrecord_files(config)
    batch_number = min(len(filenames), config.train_size) // config.batch_size
    logging.info('Total number of batches  %d' % batch_number)
    params = tf.contrib.training.HParams(
        learning_rate=config.learning_rate,
        device=config.device
    )

    run_config = tf.estimator.RunConfig(model_dir=config.checkpoint_dir)
    srcnn = get_estimator(run_config, params)
    srcnn.train(get_input_fn(filenames, config.epoch, True, config.batch_size))


def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.summaries_dir):
        os.makedirs(FLAGS.summaries_dir)
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        download_dataset(FLAGS.dataset)
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)

    setup_logging()

    # start the session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        run_training(FLAGS, sess)


if __name__ == '__main__':
    print("Start application")
    tf.app.run()
    print("Finish application")
