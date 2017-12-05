# Enhancer
Super resolution convolutional neural network(SRCNN) based on Tensorflow framework.

## Prerequisites
 * Tensorflow >=1.4.0
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * Pillow
 * Pyyaml

## Instruction
 1. Install TensorFlow(you are supposted to use virtualenv at ~/tensotflow) follow official instration https://www.tensorflow.org/install/
 2. Install python, pip, numpy, scipy, pillow and pyyaml
 3. Images should be located in data folder as follows ./data/{dataset}/{subset}/*.{extension} (e.g. ./data/cars/train/*.jpg)
 4. Preprocess images by preparing tfrecord files: ./scripts/create-tfrecords.sh
 5. Run training ./scripts/start-training-local.sh
 6. TensorBoard is available. Run from commandline: tensorboard --logdir=./summaries/{dataset}/{subset}/logs/

## Project structure
 * config.py   - configuration script
 * download.py - script to download image sets
 * tfrecords.py - script to create tfrecords 
 * model.py    - convolutional neural network model
 * main.py     - entry point 
 
## References
 * [Chao Dong's article "Image Super-Resolution Using Deep Convolutional Networks"](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) 
 * [Another Tensorflow implementation of SRCNN](https://github.com/tegg89/SRCNN-Tensorflow) 
 * [Subpixel repository](https://github.com/tetrachrome/subpixel) 
 * [Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)
 * [How to use Estimator, Experiment and Dataset to train models](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)
 * [Cloud MLE and GCE compatible TensorFlow distributed training example](https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example)
