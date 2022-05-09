#!/bin/bash
#SBATCH -J kde_nsd
#SBATCH -o kde_nsd.o
#SBATCH -t 24:00:00
#SBATCH -N 2 -n 32

module load Anaconda3/python-3.7
source activate lse

python main_nsd.py
