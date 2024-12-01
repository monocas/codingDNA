#!/bin/bash
 
#SBATCH --job-name=TAC
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH -p gpu
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:4090:1


python3 -u coding_regions.py
