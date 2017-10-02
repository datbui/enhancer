import os

import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pprint

from six.moves import xrange

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [128]")
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
        tf.train.match_filenames_once(os.path.join("./data", config.dataset, "train", "*.jpg")), num_epochs=config.epoch, shuffle=True)

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
        load_files(FLAGS)
        for epoch in xrange(FLAGS.epoch):
            image_out = sess.run(image_batch)
            print("Variable IMAGE is {}".format(image_out))


if __name__ == '__main__':
    print("start")
    tf.app.run()
