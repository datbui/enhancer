#!/usr/bin/env bash

echo 'Run testing....'
pwd
source ~/tensorflow/bin/activate
python3 main.py  --is_train=false --dataset=images_cleaned --subset=kidney_512 --image_size=512 -checkpoint_dir=/Users/dat/Projects/enhancer/checkpoint_back --output_dir=/Users/dat/Projects/results/result_03_10/output_correct_kidney
deactivate
echo 'Testing has been completed'