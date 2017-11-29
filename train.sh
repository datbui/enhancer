#!/bin/bash

echo 'Run training....'
pwd
source ~/tensorflow/bin/activate
python3 main.py
deactivate
echo 'Training has been completed'