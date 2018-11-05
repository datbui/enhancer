import os
from glob import glob

import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image

CONFIG_TXT = 'config.txt'

TFRECORD = 'tfrecord'

FILENAME = 'filename'

LR_IMAGE = 'lr_image'

INT1_IMAGE = 'int1_image'

INT2_IMAGE = 'int2_image'

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


def get_image(image_path, image_size=None, colored=True):
    image = read_image(image_path, colored)
    if image_size:
        image = do_resize(image, [image_size, image_size])
    return _pre_process(image)


def read_image(image_path, colored=True):
    image = scipy.misc.imread(image_path, flatten=(not colored), mode='RGB').astype(np.float32)
    return image


def save_output(lr_img, prediction, hr_img, path):
    h = max(hr_img.shape[0], prediction.shape[0], hr_img.shape[0])
    eh_img = do_resize(_post_process(prediction), [h, hr_img.shape[1]])
    lr_img = _post_process(lr_img)
    hr_img = _post_process(hr_img)
    out_img = np.concatenate((lr_img, eh_img, hr_img), axis=1)
    return scipy.misc.imsave(path, out_img)


def save_image(image, path, normalize=False):
    out_img = _post_process(image)
    if normalize:
        out_img = _intensity_normalization(out_img)
    return scipy.misc.imsave(path, out_img)


def save_config(target_dir, config):
    with open(os.path.join(target_dir, CONFIG_TXT), 'w+') as writer:
        writer.write(str(config.__flags))


def do_resize(x, shape):
    y = scipy.misc.imresize(x, shape, interp='bicubic')
    return y


def _unnormalize(image):
    return image * 255.


def _normalize(image):
    return image / 255.


def _pre_process(images):
    pre_processed = _normalize(np.asarray(images))
    pre_processed = pre_processed[:, :, np.newaxis] if len(pre_processed.shape) == 2 else pre_processed
    return pre_processed


def _intensity_normalization(image):
    threshold = 200
    image = np.where(image < threshold, (image + 40), image)
    mean = np.mean(np.where(image > threshold))
    image = np.where(image > threshold, (image - mean + 240), image)
    return image


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def tf_slice(tf_lr_image, dimension):
    return tf.expand_dims(tf_lr_image[:, :, :, dimension], -1)


def split_tif(input, output, size=256):
    tifs = load_files(input, 'tif')
    for tif in tifs:
        img = Image.open(tif)
        filename = os.path.basename(tif).split('.')[0]
        print(filename)
        try:
            img.seek(0)
            r = np.array(img)
            img.seek(1)
            g = np.array(img)
            img.seek(2)
            b = np.array(img)
            output_image = Image.fromarray(np.stack([r, g, b], axis=-1))
            output_image = output_image.resize([size, size])
            output_image.save(os.path.join(output, '%s.png' % filename))
        except EOFError as e:
            print(e)
            break


if __name__ == '__main__':
    print("start")
    print("finish")
