import os
import pprint
import time

import numpy as np
import tensorflow as tf
from six.moves import xrange

from download import download_dataset
from model import SRCNN
from utils import load_files, get_image, save_images, do_resize, pre_process, get_batch
from  config import FLAGS

pp = pprint.PrettyPrinter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_batch(batch_index, batch_size, data):
    return data[batch_index * batch_size:(batch_index + 1) * batch_size]


def run_training(config, session):
    input_data = load_files(os.path.join(config.data_dir, config.dataset, 'train'), 'jpg')
    batch_number = min(len(input_data), config.train_size) // config.batch_size
    print('Total number of batches  %d' % batch_number)

    step = 0
    srcnn = SRCNN(session, config.batch_size, config.image_size, config.image_resize, config.color_channels, config.learning_rate)

    if srcnn.load(config.checkpoint_dir, config.dataset):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    start_time = time.time()

    for epoch in xrange(config.epoch):

        epoch_start_time = time.time()
        for idx in xrange(0, batch_number):

            batch_files = get_batch(idx, config.batch_size, input_data)
            images = [get_image(batch_file, config.image_size, config.color_channels == 1) for batch_file in batch_files]
            resized_images = [do_resize(xx, [config.image_resize, ] * 2) for xx in images]
            input_images = pre_process(images)
            input_resized_images = pre_process(resized_images)

            err, predict = srcnn.train(input_resized_images, input_images)

            step += 1
            if step % 10 == 0:
                save_images(predict, [8, 8], './samples/outputs_%d_.jpg' % idx)
                srcnn.save(config.checkpoint_dir, config.dataset, step)
                print("Epoch: [%5d], step: [%5d], epoch_time: [%4.4f], time: [%4.4f], loss: [%.8f]" \
                      % ((epoch + 1), step, time.time() - epoch_start_time, time.time() - start_time, err))


def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        download_dataset(FLAGS.dataset)
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)

    # start the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        run_training(FLAGS, sess)


if __name__ == '__main__':
    print("start")
    tf.app.run()
