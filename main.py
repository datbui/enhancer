import os

import time
from glob import glob

from scipy.misc import imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from six.moves import xrange
from utils import *
from model import SRCNN

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [128]")
flags.DEFINE_integer("image_resize", 32, "The size of image to resize")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("epoch", 2, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
FLAGS = flags.FLAGS


def load_files(config):
    # path = os.path.join("./data", config.dataset, "train", "*.jpg")
    path = os.path.join("./data/test/*.jpg")
    files = sorted(glob(path))
    return files


def do_resize(x, shape):
    x = np.copy((x + 1.) * 127.5).astype("uint8")
    y = imresize(x, shape)
    return y

def run_training(config, session):

    input_data = load_files(config)
    batch_number = min(len(input_data), config.train_size) // config.batch_size
    print('Total number of batches  %d' % batch_number)

    counter = 0
    srcnn = SRCNN(session, config.batch_size, config.image_size, config.image_resize, config.learning_rate)

    start_time = time.time()

    for epoch in xrange(config.epoch):

        epoch_start_time = time.time()
        for idx in xrange(0, batch_number):

            print('Batch # %d' % idx)
            batch_files = get_batch(idx, config.batch_size, input_data);
            # print(' Files  %s' % batch_files)
            images = [get_image(batch_file, config.image_size, is_crop=True) for batch_file in batch_files]
            resized_images = [do_resize(xx, [config.image_resize, ] * 2) for xx in images]
            input_images = np.array(images).astype(np.float32)
            input_resized_images = np.array(resized_images).astype(np.float32)

            err, predict = srcnn.train(input_resized_images, input_images)

            counter += 1
            if counter % 10 == 0:
                print("Epoch: [%2d], step: [%2d], epoch_time: [%4.4f], time: [%4.4f], loss: [%.8f]" \
                      % ((epoch + 1), counter, time.time() - epoch_start_time, time.time() - start_time, err))

            save_images(input_images, [8, 8], './samples/inputs_small_%d_.png' % idx)
            save_images(predict, [8, 8], './samples/outputs_small_%d_.png' % idx)


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
