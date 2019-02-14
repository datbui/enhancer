import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

from config import FLAGS
from subpixel import phase_shift
from utils import tf_slice

LOG_EVERY_STEPS = 10

SUMMARY_EVERY_STEPS = 100


def model_fn(features, labels, mode, params):
    learning_rate = params.learning_rate
    devices = [('/device:%s' % d) for d in params.device.split(',')]
    with tf.name_scope('inputs'):
        for d in devices:
            with tf.device(d):
                lr_images = tf_slice(features[0], 0)
                int1_images = tf_slice(features[1], 0)
                int2_images = tf_slice(features[2], 0)
                hr_images = tf_slice(labels, 0)

    _, _, predictions = rcnn(lr_images, int1_images, int2_images, devices)

    if mode in (Modes.TRAIN, Modes.EVAL):
        with tf.name_scope('losses'):
            for d in devices:
                with tf.device(d):
                    mse = tf.losses.mean_squared_error(hr_images, predictions)
                    rmse = tf.sqrt(mse)
                    psnr = tf_psnr(mse)
                    ssim = tf_ssim(hr_images, predictions)
                    loss = 0.75 * rmse + 0.25 * (1 - ssim)
        with tf.name_scope('train'):
            for d in devices:
                with tf.device(d):
                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, tf.train.get_global_step())

    if mode in (Modes.TRAIN, Modes.EVAL):
        tf.summary.scalar('mse', mse)
        tf.summary.scalar('rmse', rmse)
        tf.summary.scalar('psnr', psnr)
        tf.summary.scalar('ssim', ssim)
        tf.summary.scalar('loss', loss)
        # tf.summary.image('predictions', predictions, max_outputs=1)

        summary_op = tf.summary.merge_all()
        summary_hook = tf.train.SummarySaverHook(save_steps=SUMMARY_EVERY_STEPS, output_dir=FLAGS.summaries_dir, summary_op=summary_op)

        logging_params = {'mse': mse, 'rmse': rmse, 'ssim': ssim, 'psnr': psnr, 'loss': loss, 'step': tf.train.get_global_step()}
        logging_hook = tf.train.LoggingTensorHook(logging_params, every_n_iter=LOG_EVERY_STEPS)

        # eval_metric_ops = {
        #     "rmse": tf.metrics.root_mean_squared_error(features, predictions)
        # }
        estimator_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mse,
            predictions=predictions,
            train_op=train_op,
            training_hooks=[logging_hook, summary_hook]
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


def conv(inputs, weights):
    return tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')


def cnn(lr_images, output_size, devices=['/device:CPU:0']):
    size = lr_images.get_shape().as_list()[1]
    ratio = int(output_size / size)
    output_channels = ratio * ratio if ratio > 1 else ratio
    filters_shape = [2, 1, 3, 2, 1]
    filters = [64, 32, 16, 8, output_channels]
    channels = lr_images.get_shape().as_list()[3]
    with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable(initializer=tf.random_normal([filters_shape[0], filters_shape[0], channels, filters[0]], stddev=1e-3), name='cnn_w1')
        w2 = tf.get_variable(initializer=tf.random_normal([filters_shape[1], filters_shape[1], filters[0], filters[1]], stddev=1e-3), name='cnn_w2')
        w3 = tf.get_variable(initializer=tf.random_normal([filters_shape[2], filters_shape[2], filters[1], filters[2]], stddev=1e-3), name='cnn_w3')
        w4 = tf.get_variable(initializer=tf.random_normal([filters_shape[3], filters_shape[3], filters[2], filters[3]], stddev=1e-3), name='cnn_w4')
        w5 = tf.get_variable(initializer=tf.random_normal([filters_shape[4], filters_shape[4], filters[3], filters[4]], stddev=1e-3), name='cnn_w5')
    with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
        b1 = tf.get_variable(initializer=tf.zeros(filters[0]), name='cnn_b1')
        b2 = tf.get_variable(initializer=tf.zeros(filters[1]), name='cnn_b2')
        b3 = tf.get_variable(initializer=tf.zeros(filters[2]), name='cnn_b3')
        b4 = tf.get_variable(initializer=tf.zeros(filters[3]), name='cnn_b4')
        b5 = tf.get_variable(initializer=tf.zeros(filters[4]), name='cnn_b5')
    with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
        for d in devices:
            with tf.device(d):
                conv1 = tf.nn.bias_add(conv(lr_images, w1), b1, name='conv_1')
                conv1r = tf.nn.leaky_relu(conv1, name='relu_1')
                conv2 = tf.nn.bias_add(conv(conv1r, w2), b2, name='conv_2')
                conv2r = tf.nn.leaky_relu(conv2, name='relu_2')
                conv3 = tf.nn.bias_add(conv(conv2r, w3), b3, name='conv_3')
                conv3r = tf.nn.leaky_relu(conv3, name='relu_3')
                conv4 = tf.nn.bias_add(conv(conv3r, w4), b4, name='conv_4')
                conv4r = tf.nn.leaky_relu(conv4, name='relu_4')
                conv5 = tf.nn.bias_add(conv(conv4r, w5), b5, name='conv_5')
                upscaled = tf.tanh(phase_shift(conv5, ratio))
                predictions = upscaled if ratio > 1 else conv5
    return predictions


def escnn(lr_images, output_size, devices=['/device:CPU:0']):
    size = lr_images.get_shape().as_list()[1]
    ratio = int(output_size / size)
    output_channels = ratio * ratio if ratio > 1 else ratio
    filters_shape = [5, 3, 3]
    filters = [64, 32, output_channels]
    channels = lr_images.get_shape().as_list()[3]
    with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable(initializer=tf.random_normal([filters_shape[0], filters_shape[0], channels, filters[0]], stddev=1e-3), name='cnn_w1')
        w2 = tf.get_variable(initializer=tf.random_normal([filters_shape[1], filters_shape[1], filters[0], filters[1]], stddev=1e-3), name='cnn_w2')
        w3 = tf.get_variable(initializer=tf.random_normal([filters_shape[2], filters_shape[2], filters[1], filters[2]], stddev=1e-3), name='cnn_w3')
    with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
        b1 = tf.get_variable(initializer=tf.zeros(filters[0]), name='cnn_b1')
        b2 = tf.get_variable(initializer=tf.zeros(filters[1]), name='cnn_b2')
        b3 = tf.get_variable(initializer=tf.zeros(filters[2]), name='cnn_b3')
    with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
        for d in devices:
            with tf.device(d):
                conv1 = tf.nn.bias_add(tf.nn.conv2d(lr_images, w1, strides=[1, 1, 1, 1], padding='SAME'), b1, name='conv_1')
                conv1r = tf.nn.relu(conv1, name='relu_1')
                conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1r, w2, strides=[1, 1, 1, 1], padding='SAME'), b2, name='conv_2')
                conv2r = tf.nn.relu(conv2, name='relu_2')
                conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2r, w3, strides=[1, 1, 1, 1], padding='SAME'), b3, name='conv_3')
                upscaled = tf.tanh(phase_shift(conv3, ratio))
                predictions = upscaled if ratio > 1 else conv3
    return predictions


def rcnn(in_images, inter1, inter2, devices=['/device:CPU:0']):
    def hidden_layer(img1, img2, img3, w1, w2, w3, wr1, wr2, b1, b2, b3, number='1'):
        h_x1 = tf.nn.leaky_relu(tf.nn.bias_add(conv(img1, w1), b1, name='h_x1_' + number))

        r_x2 = tf.image.resize_bicubic(conv(h_x1, wr1), [512, 512])
        x2 = tf.add(conv(img2, w2), r_x2)
        h_x2 = tf.nn.leaky_relu(tf.nn.bias_add(x2, b2, name='h_x2_' + number))

        r_x3 = tf.image.resize_bicubic(conv(h_x2, wr2), [1024, 1024])
        x3 = tf.add(conv(img3, w3), r_x3)
        h_x3 = tf.nn.leaky_relu(tf.nn.bias_add(x3, b3, name='h_x3_' + number))

        return h_x1, h_x2, h_x3

    channels = in_images.get_shape().as_list()[3]
    fshape = [3, 1, 2]
    fnums = [32, 16, 4]

    with tf.variable_scope('first_layer', reuse=tf.AUTO_REUSE):
        # first hidden layer
        w11 = tf.get_variable(initializer=tf.random_normal([fshape[0], fshape[0], channels, fnums[0]], stddev=1e-3), name='cnn_w11')
        w12 = tf.get_variable(initializer=tf.random_normal([fshape[0], fshape[0], channels, fnums[0]], stddev=1e-3), name='cnn_w12')
        w13 = tf.get_variable(initializer=tf.random_normal([fshape[0], fshape[0], channels, fnums[0]], stddev=1e-3), name='cnn_w13')
        wr11 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[0], fnums[0]], stddev=1e-3), name='cnn_wr11')
        wr12 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[0], fnums[0]], stddev=1e-3), name='cnn_wr12')
        # wt1 = tf.get_variable(initializer=tf.random_normal([fshape[0], fshape[0], channels, fnums[0]], stddev=1e-3), name='cnn_wt1')
        b11 = tf.get_variable(initializer=tf.zeros(fnums[0]), name='cnn_b11')
        b12 = tf.get_variable(initializer=tf.zeros(fnums[0]), name='cnn_b12')
        b13 = tf.get_variable(initializer=tf.zeros(fnums[0]), name='cnn_b13')

    with tf.variable_scope('second_layer', reuse=tf.AUTO_REUSE):
        # second hidden layer
        w21 = tf.get_variable(initializer=tf.random_normal([fshape[1], fshape[1], fnums[0], fnums[1]], stddev=1e-3), name='cnn_w21')
        w22 = tf.get_variable(initializer=tf.random_normal([fshape[1], fshape[1], fnums[0], fnums[1]], stddev=1e-3), name='cnn_w22')
        w23 = tf.get_variable(initializer=tf.random_normal([fshape[1], fshape[1], fnums[0], fnums[1]], stddev=1e-3), name='cnn_w23')
        wr21 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[1], fnums[1]], stddev=1e-3), name='cnn_wr21')
        wr22 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[1], fnums[1]], stddev=1e-3), name='cnn_wr22')
        # wt2 = tf.get_variable(initializer=tf.random_normal([fshape[1], fshape[1], fnums[0], fnums[1]], stddev=1e-3), name='cnn_wt2')
        b21 = tf.get_variable(initializer=tf.zeros(fnums[1]), name='cnn_b21')
        b22 = tf.get_variable(initializer=tf.zeros(fnums[1]), name='cnn_b22')
        b23 = tf.get_variable(initializer=tf.zeros(fnums[1]), name='cnn_b23')

    with tf.variable_scope('third_layer', reuse=tf.AUTO_REUSE):
        # third hidden layer
        w31 = tf.get_variable(initializer=tf.random_normal([fshape[2], fshape[2], fnums[1], fnums[2]], stddev=1e-3), name='cnn_w31')
        w32 = tf.get_variable(initializer=tf.random_normal([fshape[2], fshape[2], fnums[1], fnums[2]], stddev=1e-3), name='cnn_w32')
        w33 = tf.get_variable(initializer=tf.random_normal([fshape[2], fshape[2], fnums[1], fnums[2]], stddev=1e-3), name='cnn_w33')
        wr31 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[2], fnums[2]], stddev=1e-3), name='cnn_wr31')
        wr32 = tf.get_variable(initializer=tf.random_normal([1, 1, fnums[2], fnums[2]], stddev=1e-3), name='cnn_wr32')
        # wt3 = tf.get_variable(initializer=tf.random_normal([fshape[2], fshape[2], fnums[1], fnums[2]], stddev=1e-3), name='cnn_wt3')
        b31 = tf.get_variable(initializer=tf.zeros(fnums[2]), name='cnn_b31')
        b32 = tf.get_variable(initializer=tf.zeros(fnums[2]), name='cnn_b32')
        b33 = tf.get_variable(initializer=tf.zeros(fnums[2]), name='cnn_b33')

    for d in devices:
        with tf.device(d):
            l1_h1, l1_h2, l1_h3 = hidden_layer(in_images, inter1, inter2, w11, w12, w13, wr11, wr12, b11, b12, b13)

            l2_h1, l2_h2, l2_h3 = hidden_layer(l1_h1, l1_h2, l1_h3, w21, w22, w23, wr21, wr22, b21, b22, b23, '2')

            h1, h2, h3 = hidden_layer(l2_h1, l2_h2, l2_h3, w31, w32, w33, wr31, wr32, b31, b32, b33, '3')

            p1 = tf.tanh(phase_shift(h1, 2))
            p2 = tf.tanh(phase_shift(h2, 2))
            p3 = tf.tanh(phase_shift(h3, 2))

    return p1, p2, p3


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


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    """
    Compute structural similarity index metric.
    https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow

    :param img1: an input image
    :param img2: an input image
    :param cs_map:
    :param mean_metric:
    :param size:
    :param sigma:
    :return: ssim
    """
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


@DeprecationWarning
def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    """
    Compute multi-scale structural similarity index metric.
    https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow

    :param img1:
    :param img2:
    :param mean_metric:
    :param level:
    :return: msssim
    """
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
             (mssim[level - 1] ** weight[level - 1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_psnr(mse):
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


def tf_histogram_loss(img1, img2):
    """
    Calculate histogram loss between two images.

    https://pdfs.semanticscholar.org/ece3/b623232c90bb8a9021a3eb25223c4fde7069.pdf

    :param img1: an image normalized from 0 to 1
    :param img2: an image normalized from 0 to 1
    :return: MSE(hist_loss1, hist_loss2)
    """
    bins = np.math.ceil(255 / 5)
    img1 = tf.cast(img1, dtype=tf.float32)
    img2 = tf.cast(img2, dtype=tf.float32)
    value_range = [0.0, 1.0]
    step = 1.0 / bins
    hist1 = tf.histogram_fixed_width(values=img1, value_range=value_range, nbins=bins, dtype=tf.int32)
    hist2 = tf.histogram_fixed_width(values=img2, value_range=value_range, nbins=bins, dtype=tf.int32)
    hist1_loss = []
    hist2_loss = []
    for i in range(bins):
        try:
            base = i * step
            amount = tf.cast(tf.gather(hist1, i), dtype=tf.float32)
            pixels_in_range = tf.where(_tf_logic_range(img1, base, base + step), tf.div((img1 - base), step), tf.zeros(tf.shape(img1)))
            hist1_loss.append(tf.reduce_sum(tf.divide(pixels_in_range, tf.where(amount > 0, amount, 1))))
            amount = tf.cast(tf.gather(hist2, i), dtype=tf.float32)
            pixels_in_range = tf.where(_tf_logic_range(img2, base, base + step), tf.div((img2 - base), step), tf.zeros(tf.shape(img2)))
            hist2_loss.append(tf.reduce_sum(tf.divide(pixels_in_range, tf.where(amount > 0, amount, 1))))
        except ValueError as e:
            print(e)
    hist1_loss = tf.stack(hist1_loss, axis=0)
    hist2_loss = tf.stack(hist2_loss, axis=0)
    return tf.losses.mean_squared_error(hist1_loss, hist2_loss)


def _tf_logic_range(img, x, y):
    """
    Check inclusive range
    :param img:
    :param x:
    :param y:
    :return: boolean
    """
    return tf.logical_and(tf.greater_equal(img, x), tf.less_equal(img, y))


def tf_intensity_normalization(image):
    threshold = 200 / 255
    additional_1 = tf.fill(tf.shape(image), 240 / 255)
    image = tf.where(image > threshold, tf.add(tf.subtract(image, tf.reduce_mean(image)), additional_1), image)
    additional_2 = tf.fill(tf.shape(image), 15 / 255)
    image = tf.where(image < threshold, tf.add(image, additional_2), image)
    return image
