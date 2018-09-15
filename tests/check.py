from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file, parse_numpy_printoption


def main(unused_argv):
    if not FLAGS.file_name:
        print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
              "[--tensor_name=tensor_to_print] "
              "[--all_tensors] "
              "[--all_tensor_names] "
              "[--printoptions]")
        sys.exit(1)
    else:
        print_tensors_in_checkpoint_file(FLAGS.file_name, FLAGS.tensor_name,
                                         FLAGS.all_tensors, FLAGS.all_tensor_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--file_name",
        type=str,
        default="model.ckpt-302",
        help="Checkpoint filename. "
             "Note, if using Checkpoint V2 format, file_name is the "
             "shared prefix between all files in the checkpoint.")
    parser.add_argument(
        "--tensor_name",
        type=str,
        default="",
        help="Name of the tensor to inspect")
    parser.add_argument(
        "--all_tensors",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="If True, print the names and values of all the tensors.")
    parser.add_argument(
        "--all_tensor_names",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="If True, print the names of all the tensors.")
    parser.add_argument(
        "--printoptions",
        nargs="*",
        type=parse_numpy_printoption,
        help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
