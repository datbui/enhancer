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


def get_image(image_path, image_size, colored=False):
    image = scipy.misc.imread(image_path, flatten=(not colored), mode='YCbCr').astype(np.float32)
    return do_resize(image, [image_size, image_size])


def save_output(lr_img, prediction, hr_img, path):
    h = max(hr_img.shape[0], prediction.shape[0], hr_img.shape[0])
    eh_img = do_resize(_post_process(prediction), [h, hr_img.shape[1]])
    lr_img = _post_process(lr_img)
    hr_img = _post_process(hr_img)
    out_img = np.concatenate((lr_img, eh_img, hr_img), axis=1)
    return scipy.misc.imsave(path, out_img)


def save_image(image, path):
    out_img = _post_process(image)
    return scipy.misc.imsave(path, out_img)


def save_config(target_dir, config):
    with open(os.path.join(target_dir, CONFIG_TXT), 'w+') as writer:
        writer.write(str(config.__flags))


def do_resize(x, shape):
    y = scipy.misc.imresize(x, shape, interp='bicubic')
    return y


def _unnormalize(image):
    return image * 255.


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def parse_function(proto):
    features = {
        HEIGHT: tf.FixedLenFeature([], tf.int64),
        WIDTH: tf.FixedLenFeature([], tf.int64),
        DEPTH: tf.FixedLenFeature([], tf.int64),
        # TODO Reshape doesn't work, I have to put the shape here.
        HR_IMAGE: tf.FixedLenFeature((FLAGS.image_size, FLAGS.image_size, FLAGS.color_channels), tf.float32),
        LR_IMAGE: tf.FixedLenFeature((256, 256, FLAGS.color_channels), tf.float32),
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
