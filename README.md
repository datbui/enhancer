# Enhancer
Super resolution convolutional neural network(SRCNN) based on Tensorflow framework.

## Prerequisites
 * Python 3.X.X
 * Tensorflow >=1.4.0
 * Scipy>=1.0.0
 * Pillow >=4.3.0
 * Pyyaml >=3.12
 * Numpy >=1.13.3

## Instruction
 1. Install [Python 3](https://www.python.org/downloads/) 
 2. Follow the official installation process to install TensorFlow(you are supposed to use virtualenv at ~/tensotflow): https://www.tensorflow.org/install/
 3. Install python packages: pip3 install -r requirements.txt
 4. Images should be located in data folder as follows ./data/{dataset}/{subset}/*.{extension} (e.g. ./data/cars/train/*.jpg)
 5. Preprocess images by preparing tfrecord files: ./scripts/create-tfrecords.sh
 6. Run training ./scripts/start-training-local.sh
 7. TensorBoard is available. Run from commandline: tensorboard --logdir=./summaries/{dataset}/{subset}/logs/
 8. Run prediction ./scripts/start-testing-local.sh

## Project structure
 * config.py   - configuration script
 * download.py - script to download image sets
 * tfrecords.py - script to create tfrecords 
 * model.py    - convolutional neural network model
 * main.py     - entry point

## Result
Banana<br>
![orig](https://github.com/datbui/enhancer/blob/master/sample/banana.jpg)<br>
Surface of vinyl disc<br>
![orig](https://github.com/datbui/enhancer/blob/master/sample/surface_of_vinyl_disc.jpg)<br>
Velcro<br>
![orig](https://github.com/datbui/enhancer/blob/master/sample/velcro.jpg)<br>

## References
 * [Chao Dong's article "Image Super-Resolution Using Deep Convolutional Networks"](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) 
 * [Another Tensorflow implementation of SRCNN](https://github.com/tegg89/SRCNN-Tensorflow) 
 * [Subpixel repository](https://github.com/tetrachrome/subpixel) 
 * [Introduction to TensorFlow Datasets and Estimators](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html)
 * [How to use Estimator, Experiment and Dataset to train models](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)
 * [Cloud MLE and GCE compatible TensorFlow distributed training example](https://github.com/GoogleCloudPlatform/cloudml-dist-mnist-example)
