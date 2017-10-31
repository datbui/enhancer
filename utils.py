import ntpath
import os
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("data_dir", "data", "Directory name to download the train/test datasets [data]")
flags.DEFINE_string("tfrecord_dir", "tfrecords", "Directory name to store the TFRecord data [tfrecords]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [128]")
flags.DEFINE_integer("image_resize", 64, "The size of image to resize")
flags.DEFINE_integer("color_channels", 1, "The number of image color channels")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
FLAGS = flags.FLAGS


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
    post_processed.squeeze()
    return post_processed


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

def make_tfrecords(config=FLAGS):
    if not os.path.exists(config.tfrecord_dir):
        os.makedirs(config.tfrecord_dir)
    if not os.path.exists(os.path.join(config.tfrecord_dir, config.dataset)):
        os.makedirs(os.path.join(config.tfrecord_dir, config.dataset))

    files = load_files(os.path.join(config.data_dir, config.dataset, 'train'), 'jpg')
    for file in files:
        print(file)
        label_image = get_image(file, config.image_size, config.color_channels == 1)
        label_image = pre_process(label_image)
        low_quality_image = do_resize(label_image, [config.image_resize, ] * 2)
        low_quality_image = pre_process(low_quality_image)
        ipt_img = _bytes_feature(low_quality_image)
        label = _bytes_feature(label_image)
        record = tf.train.Example(features=tf.train.Features(feature={'image': ipt_img, 'label': label}))

        name = ntpath.basename(file).split('.')[0]
        tfrecord_filename = os.path.join(config.tfrecord_dir, FLAGS.dataset, name + '.tfrecord')
        with tf.python_io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(record.SerializeToString())


def _test_tfrecords(config=FLAGS):
    assert os.path.exists(config.tfrecord_dir)
    assert os.path.exists(os.path.join(config.tfrecord_dir, config.dataset))

    tfrecords_filename = load_files(os.path.join(config.tfrecord_dir, config.dataset), 'tfrecord')

    # record_iterator = tf.python_io.tf_record_iterator(path='tfrecords/celebA/000001.tfrecord')
    record_iterator = tf.python_io.tf_record_iterator(path=os.path.join(config.tfrecord_dir, config.dataset))

    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)

        image = (example.features.feature['image'].bytes_list.value[0])
        label = (example.features.feature['label'].bytes_list.value[0])
        image = np.fromstring(image, dtype=np.float64)
        image = image.reshape((config.image_resize, config.image_resize, config.color_channels))
        label = np.fromstring(label, dtype=np.float64)
        label = label.reshape((config.image_size, config.image_size, config.color_channels))
        print('Image')
        print(image.shape)
        print('Label')
        print(label.shape)
# def _test_dataset(config=FLAGS)
if __name__ == '__main__':
    print("start")
    make_tfrecords()
    # _test_tfrecords()
    print("finish")
