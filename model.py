import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

from config import FLAGS

LOG_EVERY_STEPS = 10

SUMMARY_EVERY_STEPS = 100


def model_fn(features, labels, mode, params):
    learning_rate = params.learning_rate
    filters_shape = [2, 1,  3, 2, 1]
    channels = 1
    device = '/device:%s:0' % params.device
    with tf.device(device):
        with tf.name_scope('inputs'):
            lr_images = features
            hr_images = labels

        with tf.name_scope('weights'):
            w1 = tf.Variable(tf.random_normal([filters_shape[0], filters_shape[0], channels, 64], stddev=1e-3), name='cnn_w1')
            w2 = tf.Variable(tf.random_normal([filters_shape[1], filters_shape[1], 64, 32], stddev=1e-3), name='cnn_w2')
            w3 = tf.Variable(tf.random_normal([filters_shape[2], filters_shape[2], 32, 16], stddev=1e-3), name='cnn_w3')
            w4 = tf.Variable(tf.random_normal([filters_shape[3], filters_shape[3], 16, 8], stddev=1e-3), name='cnn_w4')
            w5 = tf.Variable(tf.random_normal([filters_shape[4], filters_shape[4], 8, channels], stddev=1e-3), name='cnn_w5')

        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.zeros([64]), name='cnn_b1')
            b2 = tf.Variable(tf.zeros([32]), name='cnn_b2')
            b3 = tf.Variable(tf.zeros([16]), name='cnn_b3')
            b4 = tf.Variable(tf.zeros([8]), name='cnn_b4')
            b5 = tf.Variable(tf.zeros([channels]), name='cnn_b5')

        with tf.name_scope('predictions'):
            conv1 = tf.nn.bias_add(tf.nn.conv2d(lr_images, w1, strides=[1, 1, 1, 1], padding='SAME'), b1, name='conv_1')
            conv1r = tf.nn.relu(conv1, name='relu_1')
            conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1r, w2, strides=[1, 1, 1, 1], padding='SAME'), b2, name='conv_2')
            conv2r = tf.nn.relu(conv2, name='relu_2')
            conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2r, w3, strides=[1, 1, 1, 1], padding='SAME'), b3, name='conv_3')
            conv3r = tf.nn.relu(conv3, name='relu_3')
            conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3r, w4, strides=[1, 1, 1, 1], padding='SAME'), b4, name='conv_4')
            conv4r = tf.nn.relu(conv4, name='relu_4')
            conv5 = tf.nn.bias_add(tf.nn.conv2d(conv4r, w5, strides=[1, 1, 1, 1], padding='SAME'), b5, name='conv_5')
            predictions = conv5

        if mode in (Modes.TRAIN, Modes.EVAL):
            with tf.name_scope('losses'):
                mse = tf.losses.mean_squared_error(hr_images, predictions)
                rmse = tf.sqrt(mse)
                psnr = compute_psnr(mse)
                ssim = compute_ssim(hr_images, predictions)
                lr_hr_mse = tf.losses.mean_squared_error(hr_images, lr_images)
                lr_hr_rmse = tf.sqrt(lr_hr_mse)
                lr_hr_psnr = compute_psnr(lr_hr_mse)
                lr_hr_ssim = compute_ssim(hr_images, lr_images)
            with tf.name_scope('train'):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(rmse, tf.train.get_global_step())

    if mode in (Modes.TRAIN, Modes.EVAL):
        tf.summary.scalar('mse', mse)
        tf.summary.scalar('rmse', rmse)
        tf.summary.scalar('psnr', psnr)
        tf.summary.scalar('ssim', ssim)
        tf.summary.scalar('lr_hr_mse', lr_hr_mse)
        tf.summary.scalar('lr_hr_rmse', lr_hr_rmse)
        tf.summary.scalar('lr_hr_psnr', lr_hr_psnr)
        tf.summary.scalar('lr_hr_ssim', lr_hr_ssim)
        # tf.summary.image('predictions', predictions, max_outputs=1)

        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(save_steps=SUMMARY_EVERY_STEPS, output_dir=FLAGS.summaries_dir, summary_op=summary_op)

        logging_params = {'mse': mse, 'rmse': rmse, 'ssim': ssim, 'psnr': psnr, 'step': tf.train.get_global_step()}
        logging_hook = tf.train.LoggingTensorHook(logging_params, every_n_iter=LOG_EVERY_STEPS)

        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(features, predictions)
        }
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mse,
            predictions=predictions,
            train_op=train_op,
            training_hooks=[logging_hook, summary_hook],
            eval_metric_ops=eval_metric_ops
        )
    else:
        # mode == Modes.PREDICT:
        export_outputs = {
            'predictions': tf.estimator.export.PredictOutput({'high_res_images': predictions})
        }
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs
        )

    return estimator_spec


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    :param size:
    :param sigma:
    :return:
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def compute_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                              (sigma1_sq + sigma2_sq + C2)),
                 (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def compute_psnr(mse):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.

    Modify from https://github.com/titu1994/Image-Super-Resolution
    """
    return -10. * tf.log(mse) / tf.log(10.)
