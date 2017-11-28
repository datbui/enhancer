import os
import pprint
import logging

import tensorflow as tf

from config import FLAGS
from model import get_estimator, get_input_fn
from scripts.download import download_dataset
from utils import get_tfrecord_files, save_config
from logging.handlers import RotatingFileHandler

pp = pprint.PrettyPrinter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_training(config, session):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config)

    filenames = get_tfrecord_files(config)
    batch_number = min(len(filenames), config.train_size) // config.batch_size
    print('Total number of batches  %d' % batch_number)
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
    if not os.path.exists(os.path.join(FLAGS.summaries_dir, FLAGS.dataset, FLAGS.subset)):
        os.makedirs(os.path.join(FLAGS.summaries_dir, FLAGS.dataset, FLAGS.subset))
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        download_dataset(FLAGS.dataset)
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)

    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = RotatingFileHandler(os.path.join(FLAGS.log_dir, 'tensorflow.log'),  maxBytes=10*1024*1024)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # start the session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        run_training(FLAGS, sess)


if __name__ == '__main__':
    print("Start application")
    tf.app.run()
    print("Finish application")
