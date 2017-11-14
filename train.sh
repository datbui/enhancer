#!/bin/bash

echo 'Run training....'
pwd
source ~/tensorflow/bin/activate
python3 main.py --epoch=3000 --dataset=microscope --subset=kidney --batch_size=10 --image_size=256
deactivate
echo 'Training has been completed'