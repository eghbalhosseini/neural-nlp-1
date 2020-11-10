#!/bin/bash
#
#SBATCH --job-name=per-weights
#SBATCH --output=per-weights_%j.out
#SBATCH --error=per-weights_%j.err
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.
#SBATCH -t 14:00:00

cd /om/user/`whoami`/neural-nlp
source activate brainmodeling

export MODELNAME=distilgpt2

python neural_nlp run --benchmark Pereira2018-encoding-weights --model ${MODELNAME}  --log_level DEBUG
