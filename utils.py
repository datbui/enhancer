import os
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf
from six.moves import xrange

from config import FLAGS


def load_files(path, extension):
    path = os.path.join(path, "*." + extension)
    files = sorted(glob(path))
    return files


def _normalize(image):
    return image / 255.


def _unnormalize(image):
    return image * 255.


def get_image(image_path, image_size, is_black_white=True):
    image = scipy.misc.imread(image_path, flatten=is_black_white, mode='YCbCr').astype(np.float32)
    return do_resize(image, [image_size, image_size])


def save_output(lr_img, prediction, hr_img, path):
    h = max(hr_img.shape[0], prediction.shape[0], hr_img.shape[0])
    eh_img = do_resize(_post_process(prediction), [h, hr_img.shape[1]])
    lr_img = _post_process(lr_img)
    hr_img = _post_process(hr_img)
    out_img = np.concatenate((lr_img, eh_img, hr_img), axis=1)
    return scipy.misc.imsave(path, out_img)


def save_images(images, size, image_path):
    num_im = size[0] * size[1]
    return _imsave(images[:num_im], size, image_path)


def _imsave(images, size, path):
    return scipy.misc.imsave(path, _merge(images, size))


def _merge(images, size):
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
    pre_processed = _normalize(np.asarray(images))
    pre_processed = pre_processed[:, :, :, np.newaxis] if len(pre_processed.shape) == 3 else pre_processed
    return pre_processed


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def get_batch(batch_index, batch_size, data):
    return data[batch_index * batch_size:(batch_index + 1) * batch_size]


def _prepare_batch(batch_files, config):
    images = [get_image(batch_file, config.image_size, config.color_channels == 1) for batch_file in batch_files]
    low_quality_images = [do_resize(xx, [config.image_resize, ] * 2) for xx in images]
    low_quality_images = [do_resize(xx, [config.image_size, ] * 2) for xx in low_quality_images]
    images = pre_process(images)
    low_quality_images = pre_process(low_quality_images)
    lr_images = _float_feature(low_quality_images)
    hr_images = _float_feature(images)
    return lr_images, hr_images


def make_tfrecords(config=FLAGS):
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    if not os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset)):
        os.makedirs(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    files = load_files(os.path.join(config.data_dir, config.dataset, config.subset), config.extension)
    batch_number = min(len(files), config.train_size) // config.batch_size
    print('Batch number %d' % batch_number)
    for idx in xrange(0, batch_number):
        if idx % 10 == 0:
            print('batch %3d' % idx)

        batch_files = get_batch(idx, config.batch_size, files)
        lr_images, hr_images = _prepare_batch(batch_files, config)

        # Create a feature and record
        feature = {'lr_images': lr_images, 'hr_images': hr_images}
        record = tf.train.Example(features=tf.train.Features(feature=feature))

        # name = ntpath.basename(file).split('.')[0]
        tfrecord_filename = os.path.join(config.tfrecord_dir, config.dataset, config.subset, str(idx) + '.tfrecord')
        print(tfrecord_filename)
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(record.SerializeToString())


def parse_function(proto):
    features = {
        'hr_images': tf.FixedLenFeature((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels),
                                     tf.float32),
        'lr_images': tf.FixedLenFeature((FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels),
                                     tf.float32),
    }
    parsed_features = tf.parse_single_example(proto, features)

    lr_images = parsed_features['lr_images']
    hr_images = parsed_features['hr_images']

    return lr_images, hr_images


def _test_tfrecords(config=FLAGS):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset))

    filenames = load_files(os.path.join(config.tfrecord_dir, config.dataset), 'tfrecord')

    dataset = tf.contrib.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        while True:
            try:
                img, lbl = sess.run(next_element)
                print(img.shape)
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    print("start")
    make_tfrecords()
    # _test_tfrecords()
    print("finish")
