import ntpath
import os
import tensorflow as tf

from config import FLAGS
from utils import DEPTH, FILENAME, HEIGHT, HR_IMAGE, LR_IMAGE, TFRECORD, WIDTH, get_tfrecord_files, load_files, save_config, INT2_IMAGE, INT1_IMAGE, get_image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def parse_function(proto):
    features = {
        HEIGHT: tf.FixedLenFeature([], tf.int64),
        WIDTH: tf.FixedLenFeature([], tf.int64),
        HR_IMAGE: tf.FixedLenFeature([], tf.string),
        INT2_IMAGE: tf.FixedLenFeature([], tf.string),
        INT1_IMAGE: tf.FixedLenFeature([], tf.string),
        LR_IMAGE: tf.FixedLenFeature([], tf.string),
        FILENAME: tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(proto, features)

    lr_images = tf.reshape(tf.decode_raw(parsed_features[LR_IMAGE], tf.float32), tf.stack([256, 256, 3]))
    int1_images = tf.reshape(tf.decode_raw(parsed_features[INT1_IMAGE], tf.float32), tf.stack([512, 512, 3]))
    int2_images = tf.reshape(tf.decode_raw(parsed_features[INT2_IMAGE], tf.float32), tf.stack([1024, 1024, 3]))
    hr_images = tf.reshape(tf.decode_raw(parsed_features[HR_IMAGE], tf.float32), tf.stack([2048, 2048, 3]))
    names = parsed_features[FILENAME]

    # features = tf.stack([lr_images, int1_images, int2_images])
    # labels = hr_images

    return lr_images, int1_images, int2_images, hr_images, names


def create_tfrecords(config=FLAGS):
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    if not os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset)):
        os.makedirs(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config.tfrecord_dir, config)

    highres_files = load_files(os.path.join(config.data_dir, config.dataset, config.subset, 'highres'), config.extension)
    print("\nThere are %d files in %s dataset, subset %s\n" % (len(highres_files), config.dataset, config.subset))
    for file in highres_files:
        print(file)
        name = ntpath.basename(file).split('.')[0]
        lowres_filename = os.path.join(config.data_dir, config.dataset, config.subset, 'lowres', '%s.%s' % (name, config.extension))
        int1res_filename = os.path.join(config.data_dir, config.dataset, config.subset, 'int1res', '%s.%s' % (name, config.extension))
        int2res_filename = os.path.join(config.data_dir, config.dataset, config.subset, 'int2res', '%s.%s' % (name, config.extension))

        try:
            lr_image = get_image(lowres_filename)
            int1_image = get_image(int1res_filename)
            int2_image = get_image(int2res_filename)
            hr_image = get_image(file)
        except FileNotFoundError as e:
            tf.logging.error(e)
            continue

        # Create a feature and record
        feature = {
            HEIGHT: _int64_feature(config.image_size),
            WIDTH: _int64_feature(config.image_size),
            LR_IMAGE: _bytes_feature(tf.compat.as_bytes(lr_image.tostring())),
            INT1_IMAGE: _bytes_feature(tf.compat.as_bytes(int1_image.tostring())),
            INT2_IMAGE: _bytes_feature(tf.compat.as_bytes(int2_image.tostring())),
            HR_IMAGE: _bytes_feature(tf.compat.as_bytes(hr_image.tostring())),
            FILENAME: _bytes_feature(tf.compat.as_bytes(name))
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

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(50)
    dataset = dataset.batch(10)
    dataset = dataset.repeat(100)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        while True:
            try:
                lr_images, int1_images, int2_images, hr_images, names = sess.run(next_element)
                print('%s - %s - %s - %s - %s' % (str(lr_images.shape), str(int1_images.shape), str(int2_images.shape), str(hr_images.shape), str(names)))
            except tf.errors.OutOfRangeError as e:
                print(e)
                break


if __name__ == '__main__':
    if not os.path.exists(FLAGS.tfrecord_dir):
        os.makedirs(FLAGS.tfrecord_dir)
    print("Start %s tfrecord files" % FLAGS.tfrecord_mode)
    if FLAGS.tfrecord_mode == 'create':
        create_tfrecords()
    else:
        test_tfrecords()
    print("Finish %s tfrecord files" % FLAGS.tfrecord_mode)
