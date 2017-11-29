import logging
import logging.config
import os
import pprint
import tensorflow as tf
import yaml

from tensorflow.contrib.learn.python.learn import learn_runner
from logging.handlers import RotatingFileHandler
from config import FLAGS
from model import model_fn
from scripts.download import download_dataset
from utils import get_tfrecord_files, save_config, parse_function

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


def get_estimator(run_config=None, params=None):
    """Return the model as a Tensorflow Estimator object.
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def input_fn(filenames, epoch, shuffle, batch_size):
    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat(epoch)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels, names = iterator.get_next()
    return features, labels


def get_input_fn(filenames, num_epochs=None, shuffle=False, batch_size=1):
    return lambda: input_fn(filenames, num_epochs, shuffle, batch_size)


def experiment_fn(run_config, params):
    """Create an experiment to train and evaluate the model.
    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters
    Returns:
        (Experiment) Experiment for training the mnist model.
    """
    # You can change a subset of the run_config properties as
    run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    # # Setup data loaders
    train_input_fn = get_input_fn(params.filenames, params.epoch, True, params.batch_size)
    eval_input_fn = get_input_fn(params.filenames, 1, False, params.batch_size)

    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        # train_monitors=[train_input_hook],  # Hooks for training
        # eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=params.eval_steps  # Minibatch steps
    )
    return experiment


def run_experiment(config, session):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config)

    filenames = get_tfrecord_files(config)
    batch_number = min(len(filenames), config.train_size) // config.batch_size
    logging.info('Total number of batches  %d' % batch_number)

    params = tf.contrib.training.HParams(
        learning_rate=config.learning_rate,
        device=config.device,
        epoch=config.epoch,
        batch_size=config.batch_size,
        min_eval_frequency=100,
        train_steps=None,  # Use train feeder until its empty
        eval_steps=1,  # Use 1 step of evaluation feeder
        filenames=filenames
    )
    run_config = tf.contrib.learn.RunConfig(model_dir=config.checkpoint_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )

#TODO
# def run_prediction(config, session):
#     assert os.path.exists(config.tfrecord_dir)
#     assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))
#
#     save_config(config)
#
#     filenames = get_tfrecord_files(config)
#     batch_number = min(len(filenames), config.train_size) // config.batch_size
#     logging.info('Total number of batches  %d' % batch_number)
#
#     params = tf.contrib.training.HParams(
#         learning_rate=config.learning_rate,
#         device=config.device,
#     )
#     run_config = tf.estimator.RunConfig(model_dir=config.checkpoint_dir)
#     srcnn = get_estimator(run_config, params)
#     srcnn.train(get_input_fn(filenames, config.epoch, True, config.batch_size))



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
        run_experiment(FLAGS, sess)


if __name__ == '__main__':
    print("Start application")
    tf.app.run()
    print("Finish application")
