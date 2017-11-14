import os
import tensorflow as tf


class SRCNN:
    def __init__(self, session, batch_size, image_size, channels, learning_rate):
        self.session = session
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate

        # Build model
        self.filter_shapes = [1, 2, 1]
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
            # Prediction
            self.prediction = self.model()

        with tf.name_scope('metrics'):
            # Loss function
            self.loss = tf.losses.mean_squared_error(self.hr_images, self.prediction)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

        tf.summary.scalar('loss', self.loss)
        tf.summary.image('prediction', self.prediction)
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
        _, summary, loss, predict = self.session.run([self.train_op, self.summary_op, self.loss, self.prediction], feed_dict={self.lr_images: lr_images, self.hr_images: hr_images})
        return summary, loss, predict

    def save(self, checkpoint_dir, dataset_name, subset_name, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s_%s" % (dataset_name, subset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir, dataset_name, subset_name):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (dataset_name, subset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
