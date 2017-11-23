import os
import tensorflow as tf
import numpy as np


class SRCNN:
    def __init__(self, session, batch_size, image_size, channels, learning_rate, device):
        self.session = session
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate

        device = '/device:%s:0' % device

        # Build model
        self.filter_shapes = [1, 2, 1]

        with tf.device(device):
            with tf.name_scope('weights'):
                self.w1 = tf.Variable(tf.random_normal([self.filter_shapes[0], self.filter_shapes[0], channels, 64], stddev=1e-3), name='cnn_w1')
                self.w2 = tf.Variable(tf.random_normal([self.filter_shapes[1], self.filter_shapes[1], 64, 32], stddev=1e-3), name='cnn_w2')
                self.w3 = tf.Variable(tf.random_normal([self.filter_shapes[2], self.filter_shapes[2], 32, channels], stddev=1e-3), name='cnn_w3')

            with tf.name_scope('biases'):
                self.b1 = tf.Variable(tf.zeros([64]), name='cnn_b1')
                self.b2 = tf.Variable(tf.zeros([32]), name='cnn_b2')
                self.b3 = tf.Variable(tf.zeros([channels]), name='cnn_b3')

            with tf.name_scope('inputs'):
                self.lr_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, channels], name='low_resolution_images')
                self.hr_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, channels], name='high_resolution_images')

            with tf.name_scope('prediction'):
                self.prediction = self.model()

            with tf.name_scope('losses'):
                self.mse = tf.losses.mean_squared_error(self.hr_images, self.prediction)
                self.cosine_distance = tf.losses.cosine_distance(self.hr_images, self.prediction, -1)
                self.psnr = compute_psnr(self.mse)
                self.ssim = compute_ssim(self.hr_images, self.prediction)

            with tf.name_scope('metrics'):
                self.accurancy, _ = tf.metrics.accuracy(self.hr_images, self.prediction)
                self.recall, _ = tf.metrics.recall(self.hr_images, self.prediction)
                self.precision, _ = tf.metrics.precision(self.hr_images, self.prediction)
                session.run(tf.local_variables_initializer())

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse)

        self.saver = tf.train.Saver()

        tf.summary.scalar('mse', self.mse)
        tf.summary.scalar('psnr', self.psnr)
        tf.summary.scalar('ssim', self.ssim)
        tf.summary.scalar('cosine_distance', self.cosine_distance)
        tf.summary.scalar('accurancy', self.accurancy)
        tf.summary.scalar('recall', self.recall)
        tf.summary.scalar('precision', self.precision)

        # tf.summary.image('prediction', self.prediction)

        self.summary_op = tf.summary.merge_all()

    def model(self):
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.lr_images, self.w1, strides=[1, 1, 1, 1], padding='SAME'), self.b1, name='conv_1')
        conv1r = tf.nn.relu(conv1, name='relu_1')
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1r, self.w2, strides=[1, 1, 1, 1], padding='SAME'), self.b2, name='conv_2')
        conv2r = tf.nn.relu(conv2, name='relu_2')
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2r, self.w3, strides=[1, 1, 1, 1], padding='SAME'), self.b3, name='conv_3')
        return conv3

    def set_learning_rate(self, learning_rate):
        assert 0 < learning_rate < 1
        self.learning_rate = learning_rate

    def train(self, lr_images, hr_images):
        _, summary, loss, predict = self.session.run([self.train_op, self.summary_op, self.mse, self.prediction], feed_dict={self.lr_images: lr_images, self.hr_images: hr_images})
        return summary, loss, predict

    def save(self, checkpoint_dir, dataset_name, subset_name, step):
        model_name = "SRCNN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, dataset_name, subset_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir, dataset_name, subset_name):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, dataset_name, subset_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


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