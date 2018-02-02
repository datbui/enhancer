import ntpath
import os

import numpy as np
import tensorflow as tf

from config import FLAGS
from utils import DEPTH, FILENAME, HEIGHT, HR_IMAGE, LR_IMAGE, TFRECORD, WIDTH, do_resize, get_image, get_tfrecord_files, load_files, parse_function, save_config


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _prepare_image(file, config):
    image = get_image(file, config.image_size, config.color_channels == 1)
    image = pre_process(image)
    return image


def _normalize(image):
    return image / 255.


def pre_process(images):
    pre_processed = _normalize(np.asarray(images))
    pre_processed = pre_processed[:, :, np.newaxis] if len(pre_processed.shape) == 2 else pre_processed
    return pre_processed


def create_tfrecords(config=FLAGS):
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    if not os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset)):
        os.makedirs(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config.tfrecord_dir, config)

    highres_files = load_files(os.path.join(config.data_dir, config.dataset, config.subset, 'Highres'), config.extension)
    print(os.path.join(config.data_dir, config.dataset, config.subset, 'Highres'))
    print("\nThere are %d files in %s dataset, subset %s\n" % (len(highres_files), config.dataset, config.subset))
    for file in highres_files:
        print(file)
        name = ntpath.basename(file).split('.')[0]
        lowres_filename = os.path.join(config.data_dir, config.dataset, config.subset, 'Lowres', '%s.%s' % (name, config.extension))
        hr_image = _prepare_image(file, config)
        lr_image = _prepare_image(lowres_filename, config)

        # Create a feature and record
        feature = {
            HEIGHT: _int64_feature(config.image_size),
            WIDTH: _int64_feature(config.image_size),
            DEPTH: _int64_feature(config.color_channels),
            LR_IMAGE: _float_feature(lr_image),
            HR_IMAGE: _float_feature(hr_image),
            FILENAME: _bytes_feature(bytes(name, 'utf-8'))
        }
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        tfrecord_filename = os.path.join(config.tfrecord_dir, config.dataset, config.subset, '%s.%s' % (name, TFRECORD))
        print(tfrecord_filename)
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(record.SerializeToString())


def test_tfrecords(config=FLAGS):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset))


    filenames = get_tfrecord_files(config)

    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(10)
    dataset = dataset.repeat(100)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        while True:
            try:
                img, lbl, names = sess.run(next_element)
                print('%s\n%s\n%s ' % (str(img.shape), str(lbl.shape), str(names)))
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    print("Start %s tfrecord files" % FLAGS.tfrecord_mode)
    if FLAGS.tfrecord_mode == 'create':
        create_tfrecords()
    else:
        test_tfrecords()
    print("Finish %s tfrecord files" % FLAGS.tfrecord_mode)
