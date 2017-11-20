import os
import pprint
import time

import tensorflow as tf

from config import FLAGS
from download import download_dataset
from model import SRCNN
from utils import load_files, parse_function, save_output, save_config

pp = pprint.PrettyPrinter()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run_training(config, session):
    input_data = load_files(os.path.join(config.data_dir, config.dataset, config.subset), config.extension)
    batch_number = min(len(input_data), config.train_size) // config.batch_size
    print('Total number of batches  %d' % batch_number)

    srcnn = SRCNN(session, config.batch_size, config.image_size, config.color_channels, config.learning_rate, config.device)

    if srcnn.load(config.checkpoint_dir, config.dataset, config.subset):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    summary_writer = tf.summary.FileWriter(os.path.join(config.summaries_dir, config.dataset, config.subset, 'logs'), session.graph)

    save_config(config)

    filenames = load_files(os.path.join(config.tfrecord_dir, config.dataset, config.subset), 'tfrecord')

    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat(config.epoch)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(config.batch_size)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    session.run(iterator.initializer)

    session.run(tf.global_variables_initializer())

    step = 0
    batch = 0
    epoch = 0
    start_time = time.time()
    epoch_start_time = time.time()
    while True:
        try:
            lr_images, hr_images, names = session.run(next_element)
            summary, err, predict = srcnn.train(lr_images, hr_images)

            if step % 100 == 0:
                print("Epoch: [%5d], step: [%5d], epoch_time: [%4.4f], time: [%4.4f], loss: [%.8f]" \
                      % (epoch, step, time.time() - epoch_start_time, time.time() - start_time, err))
            step += 1
            batch += 1
            if step % batch_number == 0:
                filename = ('epoch_%d.jpg' % epoch)
                save_output(lr_images[0, :, :, :], predict[0, :, :, :], hr_images[0, :, :, :], os.path.join(config.summaries_dir, config.dataset, config.subset, filename))
                srcnn.save(config.checkpoint_dir, config.dataset, config.subset, step)
                summary_writer.add_summary(summary, step)
                epoch += 1
                batch = 0
                epoch_start_time = time.time()
        except tf.errors.OutOfRangeError:
            break



def main(_):
    pp.pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(os.path.join(FLAGS.summaries_dir, FLAGS.dataset, FLAGS.subset)):
        os.makedirs(os.path.join(FLAGS.summaries_dir, FLAGS.dataset, FLAGS.subset))
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        download_dataset(FLAGS.dataset)
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    # start the session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        run_training(FLAGS, sess)


if __name__ == '__main__':
    print("start")
    tf.app.run()
