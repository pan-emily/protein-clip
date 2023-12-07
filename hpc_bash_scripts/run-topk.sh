#!/bin/bash

#SBATCH --job-name=run-topk-visualization-512
#SBATCH --time=10:00:00
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH --mail-user=jdao@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH -o "/groups/mlprojects/protein-clip-pjt/outs/%j-%x.txt"

MINICONDA_PATH=/groups/mlprojects/protein-clip-pjt/miniconda3
ENV_NAME=protein-clip

source $MINICONDA_PATH/etc/profile.d/conda.sh

conda activate $ENV_NAME

cd protein-clip

export PYTHONUNBUFFERED=TRUE
python topk.py

conda deactivate