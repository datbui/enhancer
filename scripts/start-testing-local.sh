#!/usr/bin/env bash

echo 'Run testing....'
pwd
source ~/tensorflow/bin/activate
python3 main.py --is_train=false --subset=test
deactivate
echo 'Testing has been completed'