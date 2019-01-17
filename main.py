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
from model import model_fn, rcnn
from tfrecord import parse_function
from utils import get_tfrecord_files, save_config, save_image, save_output, tf_slice

PREDICTION = 'prediction'

INT1 = 'int1'

INT2 = 'int2'

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
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    if shuffle:
        dataset = dataset.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    prefetch_batch_size = batch_size
    dataset = dataset.prefetch(prefetch_batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs, int1_inputs, int2_inputs, labels, names = iterator.get_next()
    features = [inputs, int1_inputs, int2_inputs]
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
        min_eval_frequency=100,
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

    dataset = tf.data.TFRecordDataset(files, buffer_size=10)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    tf_next_element = iterator.get_next()

    tf_lr_image, tf_int1_image, tf_int2_image, tf_hr_image_tensor, tf_name = tf_next_element
    tf_re_image = tf.image.resize_images(tf_lr_image, [2048, 2048])
    # tf_initial_mse = tf.losses.mean_squared_error(tf_hr_image_tensor, tf_re_image)
    # tf_initial_rmse = tf.sqrt(tf_initial_mse)
    # tf_initial_psnr = tf.image.psnr(tf_hr_image_tensor, tf_re_image, max_val=1.0)
    # tf_initial_ssim = tf.image.ssim(tf_hr_image_tensor, tf_re_image, max_val=1.0)
    # tf_initial_msssim = tf.image.ssim_multiscale(tf_hr_image_tensor, tf_re_image, max_val=1.0)
    # tf_initial_params = [tf_initial_rmse, tf_initial_psnr, tf_initial_ssim, tf_initial_msssim]

    r1, r2, r3 = rcnn(tf_slice(tf_lr_image, 0), tf_slice(tf_int1_image, 0), tf_slice(tf_int2_image, 0))
    g1, g2, g3 = rcnn(tf_slice(tf_lr_image, 1), tf_slice(tf_int1_image, 1), tf_slice(tf_int2_image, 1))
    b1, b2, b3 = rcnn(tf_slice(tf_lr_image, 2), tf_slice(tf_int1_image, 2), tf_slice(tf_int2_image, 2))
    # tf_prediction = rcnn(slice(tf_lr_image, 2), slice(tf_int1_image, 2), slice(tf_int2_image, 2))
    tf_prediction1 = tf.stack([tf.squeeze(r1, axis=3), tf.squeeze(g1, axis=3), tf.squeeze(b1, axis=3)], axis=3)
    tf_prediction2 = tf.stack([tf.squeeze(r2, axis=3), tf.squeeze(g2, axis=3), tf.squeeze(b2, axis=3)], axis=3)
    tf_prediction3 = tf.stack([tf.squeeze(r3, axis=3), tf.squeeze(g3, axis=3), tf.squeeze(b3, axis=3)], axis=3)
    tf_prediction = (tf_prediction1, tf_prediction2, tf_prediction3)
    tf.initialize_all_variables().run()

    tf_predicted_mse = tf.losses.mean_squared_error(tf_hr_image_tensor, tf_prediction)
    tf_predicted_rmse = tf.sqrt(tf_predicted_mse)
    tf_predicted_psnr = tf.image.psnr(tf_hr_image_tensor, tf_prediction, max_val=1.0)
    tf_predicted_ssim = tf.image.ssim(tf_hr_image_tensor, tf_prediction, max_val=1.0)
    tf_predicted_msssim = tf.image.ssim_multiscale(tf_hr_image_tensor, tf_prediction, max_val=1.0)
    tf_predicted_params = [tf_predicted_rmse, tf_predicted_psnr, tf_predicted_ssim, tf_predicted_msssim]

    load(session, config.checkpoint_dir)

    count = 1
    try:
        params_file = open('metrics.csv', 'w+')
        writer = csv.writer(params_file)
        # writer.writerows([['filename', 'initial_rmse', 'rmse', 'initial_psnr', 'psnr', 'initial_ssim', 'ssim', 'initial_msssim', 'msssim', 'initial_nmi', 'nmi', 'initial_wsnr', 'wsnr', 'initial_ifc', 'ifc', 'initial_nqm', 'nqm']])

        writer.writerows([['filename', 'rmse', 'psnr', 'ssim', 'msssim']])

        while True:
            try:
                next_element, re_image, prediction, predicted_params = session.run([tf_next_element, tf_re_image, tf_prediction, tf_predicted_params])

                (rmse, psnr, ssim, msssim) = predicted_params
                (lr_image, _, _, hr_image, name) = next_element
                name = str(name[0]).replace('b\'', '').replace('\'', '')

                prediction1 = np.squeeze(prediction1)
                prediction2 = np.squeeze(prediction2)
                prediction3 = np.squeeze(prediction3)
                re_image = np.squeeze(re_image)
                hr_image = np.squeeze(hr_image)

                # initial_nmi = normalized_mutual_info_score(hr_image.flatten(), re_image.flatten())
                # nmi = normalized_mutual_info_score(hr_image.flatten(), prediction.flatten())

                # initial_wsnr = wsnr(hr_image, re_image)
                # _wsnr = wsnr(hr_image, prediction)

                # initial_ifc = pbvif(hr_image, re_image)
                # _ifc = pbvif(hr_image, prediction)

                # initial_nqm = nqm(hr_image, re_image)
                # _nqm = nqm(hr_image, prediction)

                writer.writerows([[name, rmse, np.squeeze(psnr), np.squeeze(ssim), np.squeeze(msssim)]])
                save_image(image=prediction1, path=os.path.join(config.output_dir, INT1, '%s.jpg' % name))
                save_image(image=prediction2, path=os.path.join(config.output_dir, INT2, '%s.jpg' % name))
                save_image(image=prediction3, path=os.path.join(config.output_dir, PREDICTION, '%s.jpg' % name))
                save_image(image=re_image, path=os.path.join(config.output_dir, LOW_RESOLUTION, '%s.jpg' % name))
                save_image(image=hr_image, path=os.path.join(config.output_dir, HIGH_RESOLUTION, '%s.jpg' % name))
                save_output(lr_img=re_image, prediction=prediction3, hr_img=hr_image, path=os.path.join(config.output_dir, '%s.jpg' % name))

                logging.info("Enhance resolution for %3.0d %s" % (count, name))
                count = count + 1
            except tf.errors.OutOfRangeError as e:
                logging.error(e)
                break
    except KeyboardInterrupt as e:
        print("Cancel by user")
    finally:
        params_file.close()


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    setup_logging()

    # start the session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        if FLAGS.is_train:
            if not os.path.exists(FLAGS.checkpoint_dir):
                os.makedirs(FLAGS.checkpoint_dir)
            if not os.path.exists(FLAGS.summaries_dir):
                os.makedirs(FLAGS.summaries_dir)
            run_training(sess)
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(os.path.join(FLAGS.output_dir, PREDICTION))
                os.makedirs(os.path.join(FLAGS.output_dir, INT1))
                os.makedirs(os.path.join(FLAGS.output_dir, INT2))
                os.makedirs(os.path.join(FLAGS.output_dir, LOW_RESOLUTION))
                os.makedirs(os.path.join(FLAGS.output_dir, HIGH_RESOLUTION))
            run_testing(sess)


if __name__ == '__main__':
    print("Start application")
    tf.app.run()
    print("Finish application")
