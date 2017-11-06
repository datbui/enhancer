import os
import tensorflow as tf


class SRCNN:
    def __init__(self, session, batch_size, image_size, image_resize, color_channels, learning_rate):
        self.session = session
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_resize = image_resize

        self.learning_rate = learning_rate

        # TODO rewrite generate function
        t_vars = tf.trainable_variables()

        # Build model
        self.filter_shapes = [9, 1, 5]
        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.filter_shapes[0], self.filter_shapes[0], color_channels, 64], stddev=1e-3), name='cnn_w1'),
            'w2': tf.Variable(tf.random_normal([self.filter_shapes[1], self.filter_shapes[1], 64, 32], stddev=1e-3), name='cnn_w2'),
            'w3': tf.Variable(tf.random_normal([self.filter_shapes[2], self.filter_shapes[2], 32, color_channels], stddev=1e-3), name='cnn_w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='cnn_b1'),
            'b2': tf.Variable(tf.zeros([32]), name='cnn_b2'),
            'b3': tf.Variable(tf.zeros([color_channels]), name='cnn_b3')
        }

        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, color_channels], name='low_resolution_images')
        self.lr_images = self.inputs
        self.hr_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, color_channels], name='high_resolution_images')
        label_size = self.image_size - sum(self.filter_shapes) + len(self.filter_shapes)
        self.label_images = tf.image.resize_images(self.hr_images, [label_size, label_size], tf.image.ResizeMethod.BICUBIC)

        # Prediction
        self.h = self.model()

        # Loss function
        self.j = tf.losses.mean_squared_error(self.label_images, self.h)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.j)

        session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def model(self):
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.lr_images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID'), self.biases['b1'], name='add_bias_1')
        conv1r = tf.nn.relu(conv1)
        conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1r, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID'), self.biases['b2'], name='add_bias_2')
        conv2r = tf.nn.relu(conv2)
        conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2r, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID'), self.biases['b3'], name='add_bias_3')
        return conv3

    def set_learning_rate(self, learning_rate):
        assert 0 < learning_rate < 1
        self.learning_rate = learning_rate

    def train(self, lr_images, hr_images):
        _, loss, predict = self.session.run([self.train_op, self.j, self.h], feed_dict={self.inputs: lr_images, self.hr_images: hr_images})
        return loss, predict

    def save(self, checkpoint_dir, dataset_name, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % (dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.session, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir, dataset_name):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
