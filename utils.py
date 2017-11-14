import ntpath
import os
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf

from config import FLAGS

CONFIG_TXT = 'config.txt'

TFRECORD = '.tfrecord'

FILENAME = 'filename'

LR_IMAGE = 'lr_image'

HR_IMAGE = 'hr_image'


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


def save_config(config):
    with open(os.path.join(config.tfrecord_dir, config.dataset, config.subset, CONFIG_TXT), 'w+') as writer:
        writer.write(str(config.__flags))


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
    pre_processed = pre_processed[:, :, np.newaxis] if len(pre_processed.shape) == 2 else pre_processed
    return pre_processed


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def get_batch(batch_index, batch_size, data):
    return data[batch_index * batch_size:(batch_index + 1) * batch_size]


def _prepare_image(file, config):
    image = get_image(file, config.image_size, config.color_channels == 1)
    low_quality_image = do_resize(image, [config.image_resize, ] * 2)
    low_quality_image = do_resize(low_quality_image, [config.image_size, ] * 2)
    image = pre_process(image)
    low_quality_image = pre_process(low_quality_image)
    lr_image = _float_feature(low_quality_image)
    hr_image = _float_feature(image)
    return lr_image, hr_image


def make_tfrecords(config=FLAGS):
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    if not os.path.exists(os.path.join(config.tfrecord_dir, config.dataset, config.subset)):
        os.makedirs(os.path.join(config.tfrecord_dir, config.dataset, config.subset))

    save_config(config)

    files = load_files(os.path.join(config.data_dir, config.dataset, config.subset), config.extension)
    for file in files:
        print(file)
        lr_image, hr_image = _prepare_image(file, config)
        name = ntpath.basename(file).split('.')[0]
        # Create a feature and record
        feature = {LR_IMAGE: lr_image, HR_IMAGE: hr_image, FILENAME: _bytes_feature(bytes(name, 'utf-8'))}
        record = tf.train.Example(features=tf.train.Features(feature=feature))


        tfrecord_filename = os.path.join(config.tfrecord_dir, config.dataset, config.subset, name + TFRECORD)
        print(tfrecord_filename)
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(record.SerializeToString())




def parse_function(proto):
    features = {
        HR_IMAGE: tf.FixedLenFeature((FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels), tf.float32),
        LR_IMAGE: tf.FixedLenFeature((FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels), tf.float32),
        FILENAME: tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(proto, features)

    lr_images = parsed_features[LR_IMAGE]
    hr_images = parsed_features[HR_IMAGE]
    name = parsed_features[FILENAME]

    return lr_images, hr_images, name


def _test_tfrecords(config=FLAGS):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset))

    filenames = load_files(os.path.join(config.tfrecord_dir, config.dataset, config.subset), 'tfrecord')

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
                img, _, lbl = sess.run(next_element)
                print(str(img.shape) + str(lbl))
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    print("start")
    make_tfrecords()
    # _test_tfrecords()
    print("finish")
