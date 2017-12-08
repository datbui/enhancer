import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("dataset", "images", "The name of dataset [celebA, mnist, lsun ...]")
flags.DEFINE_string("subset", "train", "The name of subset [train, validation, test]")
flags.DEFINE_string("extension", "jpg", "The file extension [tif, jpg....]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summaries_dir", "summaries", "Directory name to save training summaries[summaries]")
flags.DEFINE_string("data_dir", "data", "Directory name to download the train/test datasets [data]")
flags.DEFINE_string("tfrecord_dir", "tfrecords", "Directory name to store the TFRecord data [tfrecords]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [256]")
flags.DEFINE_integer("image_resize", 128, "The size of image to resize [128]")
flags.DEFINE_integer("color_channels", 1, "The number of image color channels")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("epoch", 1500, "Epoch to train [1000]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_string("device", 'CPU', "The device: CPU or GPU [CPU]")
flags.DEFINE_string("tfrecord_mode", 'test', "Mode to create/test tfrecord files. Default is [test]")
flags.DEFINE_string("log_dir", "logs", "Directory name to store logs [logs]")
flags.DEFINE_bool("is_train", "true", "Train or test mode")
FLAGS = flags.FLAGS
