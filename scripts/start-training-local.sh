#!/bin/bash

echo 'Run training....'
pwd
source ~/tensorflow/bin/activate
python3 main.py --dataset=images_cleaned --subset=breast_512 --learning_rate=0.001 --image_size=512 --epoch=1000
deactivate
echo 'Training has been completed'