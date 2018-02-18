"""
Modification of https://github.com/stanfordnlp/treelstm/blob/master/scripts/download.py
Copy from https://github.com/tetrachrome/subpixel/blob/master/download.py
Downloads the following:
- Celeb-A dataset
- LSUN dataset
- MNIST dataset
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import zipfile

from six.moves import urllib

parser = argparse.ArgumentParser(description='Download dataset for DCGAN.')
parser.add_argument('--datasets', metavar='N', type=str, nargs='+',
                    help='name of dataset to download [celebA, lusn, mnist]')


def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
                  ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath


def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)


def download_celeb_a(dirpath):
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637
    data_dir = 'celebA'
    if os.path.exists(os.path.join(dirpath, data_dir)):
        print('Found Celeb-A - skip')
        return
    url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1&pv=1'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)

    # now split data into train/valid/test
    train_dir = os.path.join(dirpath, zip_dir, 'train')
    valid_dir = os.path.join(dirpath, zip_dir, 'valid')
    test_dir = os.path.join(dirpath, zip_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    zip_path = os.path.join(dirpath, zip_dir)
    for i in range(NUM_EXAMPLES):
        image_filename = "{:06d}.jpg".format(i + 1)
        candidate_file = os.path.join(zip_path, image_filename)
        if os.path.exists(candidate_file):
            if i < TRAIN_STOP:
                dest_dir = train_dir
            elif i < VALID_STOP:
                dest_dir = valid_dir
            else:
                dest_dir = test_dir
            dest_file = os.path.join(dest_dir, image_filename)
            os.rename(candidate_file, dest_file)

    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))


def _list_categories(tag):
    url = 'http://lsun.cs.princeton.edu/htbin/list.cgi?tag=' + tag
    f = urllib.request.urlopen(url)
    return json.loads(f.read())


def _download_lsun(out_dir, category, set_name, tag):
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    print(url)
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = os.path.join(out_dir, out_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading', category, set_name, 'set')
    subprocess.call(cmd)


def download_lsun(dirpath):
    data_dir = os.path.join(dirpath, 'lsun')
    if os.path.exists(data_dir):
        print('Found LSUN - skip')
        return
    else:
        os.mkdir(data_dir)

    tag = 'latest'
    # categories = _list_categories(tag)
    categories = ['bedroom']

    for category in categories:
        _download_lsun(data_dir, category, 'train', tag)
        _download_lsun(data_dir, category, 'val', tag)
    _download_lsun(data_dir, '', 'test', tag)


def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
                  't10k-labels-idx3-ubyte.gz']
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip -d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)


def prepare_data_dir(path='./data'):
    if not os.path.exists(path):
        os.mkdir(path)


def download_dataset(dataset_names):
    prepare_data_dir()
    if 'celebA' in dataset_names:
        download_celeb_a('./data')
    if 'lsun' in dataset_names:
        download_lsun('./data')
    if 'mnist' in dataset_names:
        download_mnist('./data')


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.datasets:
        raise Exception(" [!] You need to specify the name of datasets to download")

    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        download_dataset(FLAGS.dataset)
    download_dataset(args.datasets)
