import os
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf

from config import FLAGS

CONFIG_TXT = 'config.txt'

TFRECORD = 'tfrecord'

FILENAME = 'filename'

LR_IMAGE = 'lr_image'

HR_IMAGE = 'hr_image'

HEIGHT = 'height'

WIDTH = 'width'

DEPTH = 'depth'


def load_files(path, extension):
    path = os.path.join(path, "*.%s" % extension)
    files = sorted(glob(path))
    return files


def get_tfrecord_files(config):
    return load_files(os.path.join(config.tfrecord_dir, config.dataset, config.subset), TFRECORD)


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


def _normalize(image):
    return image / 255.


def _unnormalize(image):
    return image * 255.


def pre_process(images):
    pre_processed = _normalize(np.asarray(images))
    pre_processed = pre_processed[:, :, np.newaxis] if len(pre_processed.shape) == 2 else pre_processed
    return pre_processed


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def parse_function(proto):
    features = {
        HEIGHT: tf.FixedLenFeature([], tf.int64),
        WIDTH: tf.FixedLenFeature([], tf.int64),
        DEPTH: tf.FixedLenFeature([], tf.int64),
        # TODO Reshape doesn't work, I have to put dimension here.
        HR_IMAGE: tf.FixedLenFeature((FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels), tf.float32),
        LR_IMAGE: tf.FixedLenFeature((FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels), tf.float32),
        FILENAME: tf.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.parse_single_example(proto, features)

    lr_images = parsed_features[LR_IMAGE]
    hr_images = parsed_features[HR_IMAGE]
    name = parsed_features[FILENAME]

    return lr_images, hr_images, name


if __name__ == '__main__':
    print("start")
    print("finish")
