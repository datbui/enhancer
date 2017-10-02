import os

import time
from glob import glob

from scipy.misc import imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pprint

from six.moves import xrange
from utils import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [128]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("epoch", 2, "Epoch to train [25]")
FLAGS = flags.FLAGS


def read_image(filename_queue):
    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    image_file = image_reader.read(filename_queue)

    return image_file


def load_files(config):
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once(os.path.join("./data", config.dataset, "train", "*.jpg")),
        num_epochs=config.epoch, shuffle=True)

    return filename_queue


def input_pipeline(config):
    filename_queue = load_files(config)
    image = read_image(filename_queue)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * config.batch_size
    image_batch = tf.train.shuffle_batch(image, batch_size=config.batch_size, capacity=capacity,
                                         min_after_dequeue=min_after_dequeue)

    return image_batch


def doresize(x, shape):
    x = np.copy((x + 1.) * 127.5).astype("uint8")
    y = imresize(x, shape)
    return y


def just_run(config):
    input_size = 32;
    data = sorted(glob(os.path.join("./data", config.dataset, "train", "*.jpg")))
    batch_idxs = min(len(data), config.train_size) // config.batch_size
    for idx in xrange(0, batch_idxs):
        print(' IDX  %d' % idx)
        batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
        print(' Files  %s' % batch_files)
        batch = [get_image(batch_file, config.image_size, is_crop=True) for batch_file in batch_files]
        input_batch = [doresize(xx, [input_size, ] * 2) for xx in batch]
        batch_images = np.array(batch).astype(np.float32)
        batch_inputs = np.array(input_batch).astype(np.float32)
        save_images(batch_images, [8, 8], './samples/inputs_small_%d_.png' % idx)


def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    print("Init")
    # setup the variable initialisation
    init_op = tf.global_variables_initializer()
    image_batch = input_pipeline(FLAGS)

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        start_time = time.time()
        just_run(FLAGS)
        print("Time ")
        print(time.time() - start_time)


if __name__ == '__main__':
    print("start")
    tf.app.run()
