#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH -o output_filename_%j.txt
module load usermods
module load user/cuda
module load user/python-site

python3 main.py --device=GPU