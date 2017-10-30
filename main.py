import os
import pprint
import time
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf
from six.moves import xrange

from model import SRCNN

pp = pprint.PrettyPrinter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [128]")
flags.DEFINE_integer("image_resize", 64, "The size of image to resize")
flags.DEFINE_integer("color_channels", 1, "The number of image color channels")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
FLAGS = flags.FLAGS


def load_files(config):
    # path = os.path.join("./data", config.dataset, "train", "*.jpg")
    path = os.path.join("./data/test/*.jpg")
    files = sorted(glob(path))
    return files


def normalize(image):
    return np.array(image) / 255.


def unnormalize(image):
    return image * 255.


def get_image(image_path, image_size, is_black_white=True):
    image = scipy.misc.imread(image_path, flatten=is_black_white, mode='YCbCr').astype(np.float32)
    return do_resize(image, [image_size, image_size])


def save_images(images, size, image_path):
    num_im = size[0] * size[1]
    return imsave(images[:num_im], size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def do_resize(x, shape):
    y = scipy.misc.imresize(x, shape, interp='bicubic')
    return y


def pre_process(images):
    pre_processed = normalize(images)
    pre_processed = pre_processed[:, :, :, np.newaxis] if len(pre_processed.shape) == 3 else pre_processed
    return pre_processed


def post_process(images):
    post_processed = unnormalize(images)
    post_processed.squeeze()
    return post_processed


def run_training(config, session):
    input_data = load_files(config)
    batch_number = min(len(input_data), config.train_size) // config.batch_size
    print('Total number of batches  %d' % batch_number)

    counter = 0
    srcnn = SRCNN(session, config.batch_size, config.image_size, config.image_resize, config.color_channels,
                  config.learning_rate)

    start_time = time.time()

    for epoch in xrange(config.epoch):

        epoch_start_time = time.time()
        for idx in xrange(0, batch_number):

            batch_files = get_batch(idx, config.batch_size, input_data);
            images = [get_image(batch_file, config.image_size) for batch_file in batch_files]
            resized_images = [do_resize(xx, [config.image_resize, ] * 2) for xx in images]
            input_images = pre_process(images)
            input_resized_images = pre_process(resized_images)

            err, predict = srcnn.train(input_resized_images, input_images)

            counter += 1
            if counter % 10 == 0:
                print("Epoch: [%2d], step: [%2d], epoch_time: [%4.4f], time: [%4.4f], loss: [%.8f]" \
                      % ((epoch + 1), counter, time.time() - epoch_start_time, time.time() - start_time, err))

            save_images((predict), [8, 8], './samples/outputs_%d_.jpg' % idx)


def get_batch(batch_index, batch_size, data):
    return data[batch_index * batch_size:(batch_index + 1) * batch_size]


def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # start the session
    with tf.Session() as sess:
        run_training(FLAGS, sess)


if __name__ == '__main__':
    print("start")
    tf.app.run()
