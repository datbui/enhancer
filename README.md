# Enhancer
Super resolution convolutional neural network(SRCNN) based on Tensorflow framework.

## Prerequisites
 * Tensorflow >=1.3.0
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)

## Instruction
 1. Install TensorFlow(you are supposted to use virtualenv at ~/tensotflow) follow official instration https://www.tensorflow.org/install/
 2. Install python, pip, numpy, scipy, pillow
 3. Images should be located in data folder as follows ./data/{dataset}/{subset}/*.{extension} (e.g. ./data/cars/train/*.jpg)
 4. Preprocess images by preparing tfrecord files: pythorn3 utitls.py --dataset={dataset} --subset={subset}
 5. Run training ./train.sh
 6. TensorBoard is available. Run from commandline: tensorboard --logdir=./summaries/{dataset}/{subset}/logs/

## Project structure
 * config.py   - configuration script
 * download.py - downloading image samples script
 * nodel.py    - convolutional neural network model
 * main.py     - entry point 
 
## References
 * [Chao Dong's article "Image Super-Resolution Using Deep Convolutional Networks"](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) 
 * [Another Tensorflow implementation of SRCNN](https://github.com/tegg89/SRCNN-Tensorflow) 
 * [Subpixel repository](https://github.com/tetrachrome/subpixel) 
