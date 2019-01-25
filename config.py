import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("dataset", "images_cleaned", "The name of dataset [celebA, mnist, lsun ...]")
flags.DEFINE_string("subset", "train", "The name of subset [train, validation, test]")
flags.DEFINE_string("extension", "png", "The file extension [tif, jpg....]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("summaries_dir", "summaries", "Directory name to save training summaries[summaries]")
flags.DEFINE_string("log_dir", "logs", "Directory name to store logs [logs]")
flags.DEFINE_string("output_dir", "outputs", "Directory name to store output images [outputs]")
flags.DEFINE_string("data_dir", "data", "Directory name to download the train/test datasets [data]")
flags.DEFINE_string("tfrecord_dir", "tfrecords", "Directory name to store the TFRecord data [tfrecords]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("image_size", 512, "The size of image to use")
flags.DEFINE_integer("epoch", 1500, "Epoch to train [1000]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_string("device", 'CPU:0', "The device: CPU or GPU [CPU:0]")
flags.DEFINE_string("tfrecord_mode", 'test', "Mode to create/test tfrecord files. Default is [test]")
flags.DEFINE_bool("is_train", "true", "Train or test mode")
FLAGS = flags.FLAGS
