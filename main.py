import csv
import logging
import logging.config
import os
import pprint
from logging.handlers import RotatingFileHandler

import numpy as np
import tensorflow as tf
import yaml
from tensorflow.contrib.learn.python.learn import learn_runner

from config import FLAGS
from download import download_dataset
from model import model_fn, srcnn, tf_psnr, tf_ssim
from utils import get_tfrecord_files, parse_function, save_config, save_image, save_output

PREDICTION = 'prediction'

LOW_RESOLUTION = 'low_resolution'

HIGH_RESOLUTION = 'high_resolution'

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
    train_input_fn = get_input_fn(params.train_files, params.epoch, True, params.batch_size)

    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=train_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        # train_monitors=[train_input_hook],  # Hooks for training
        # eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=params.eval_steps  # Minibatch steps
    )
    return experiment


def run_training(session, config=FLAGS):
    save_config(config.summaries_dir, config)

    train_files = get_tfrecord_files(config)
    batch_number = len(train_files) // config.batch_size
    logging.info('Total number of batches  %d' % batch_number)

    params = tf.contrib.training.HParams(
        learning_rate=config.learning_rate,
        pkeep_conv=0.75,
        device=config.device,
        epoch=config.epoch,
        batch_size=config.batch_size,
        min_eval_frequency=500,
        train_steps=None,  # Use train feeder until its empty
        eval_steps=1,  # Use 1 step of evaluation feeder
        train_files=train_files
    )
    run_config = tf.contrib.learn.RunConfig(model_dir=config.checkpoint_dir)
    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train",  # What to run
        hparams=params  # HParams
    )


def load(session, checkpoint_dir):
    logging.info(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        save_path = os.path.join(checkpoint_dir, ckpt_name)
        logging.info(save_path)
        tf.train.Saver().restore(session, save_path)
        return True
    else:
        return False


def run_testing(session, config=FLAGS):
    files = get_tfrecord_files(config)
    logging.info('Total number of files  %d' % len(files))

    dataset = tf.contrib.data.TFRecordDataset(files)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    session.run(iterator.initializer)

    (tf_lr_image, tf_hr_image_tensor, _) = next_element
    tf_initial_mse = tf.losses.mean_squared_error(tf_hr_image_tensor, tf_lr_image)
    tf_initial_rmse = tf.sqrt(tf_initial_mse)
    tf_initial_psnr = tf_psnr(tf_initial_mse)
    tf_initial_ssim = tf_ssim(tf_hr_image_tensor, tf_lr_image)

    tf_prediction = srcnn(tf_lr_image)
    # tf_prediction = tf_intensity_normalization(tf_prediction)
    predicted_mse = tf.losses.mean_squared_error(tf_hr_image_tensor, tf_prediction)
    predicted_rmse = tf.sqrt(predicted_mse)
    predicted_psnr = tf_psnr(predicted_mse)
    predicted_ssim = tf_ssim(tf_hr_image_tensor, tf_prediction)

    load(session, config.checkpoint_dir)

    params_file = open('metrics.csv', 'w+')
    writer = csv.writer(params_file)
    writer.writerows([['filename', 'initial_rmse', 'rmse', 'initial_psnr', 'psnr', 'initial_ssim', 'ssim']])

    while True:
        try:
            initial_rmse, initial_psnr, initial_ssim = session.run([tf_initial_rmse, tf_initial_psnr, tf_initial_ssim])
            rmse, psnr, ssim = session.run([predicted_rmse, predicted_psnr, predicted_ssim])
            (lr_image, hr_image, name), prediction = session.run([next_element, tf_prediction])
            prediction = np.squeeze(prediction)
            name = str(name[0]).replace('b\'', '').replace('\'', '')
            logging.info('Enhance resolution for %s' % name)
            writer.writerows([[name, initial_rmse, rmse, initial_psnr, psnr, initial_ssim, ssim]])
            save_image(image=prediction, path=os.path.join(config.output_dir, PREDICTION, '%s.jpg' % name))
            save_image(image=lr_image, path=os.path.join(config.output_dir, LOW_RESOLUTION, '%s.jpg' % name))
            save_image(image=hr_image, path=os.path.join(config.output_dir, HIGH_RESOLUTION, '%s.jpg' % name))
            save_output(lr_img=lr_image, prediction=prediction, hr_img=hr_image, path=os.path.join(config.output_dir, '%s.jpg' % name))
        except tf.errors.OutOfRangeError:
            break

    params_file.close()


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    setup_logging()

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    # start the session
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(os.path.join(FLAGS.output_dir, PREDICTION))
                os.makedirs(os.path.join(FLAGS.output_dir, LOW_RESOLUTION))
                os.makedirs(os.path.join(FLAGS.output_dir, HIGH_RESOLUTION))
            if not os.path.exists(FLAGS.summaries_dir):
                os.makedirs(FLAGS.summaries_dir)
            run_training(sess)
        else:
            run_testing(sess)


if __name__ == '__main__':
    print("Start application")
    tf.app.run()
    print("Finish application")
