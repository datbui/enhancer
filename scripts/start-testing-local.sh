#!/usr/bin/env bash

echo 'Run testing....'
pwd
source ~/tensorflow/bin/activate
python3 main.py --is_train=false --dataset=images_cleaned
deactivate
echo 'Testing has been completed'