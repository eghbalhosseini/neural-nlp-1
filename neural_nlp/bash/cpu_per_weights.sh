#!/bin/bash
#
#SBATCH --job-name=per-weights
#SBATCH --output=per-weights_%j.out
#SBATCH --error=per-weights_%j.err
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH -t 28:00:00

source activate brainmodeling
cd /om/user/`whoami`/neural-nlp

python neural_nlp run --benchmark Pereira2018-encoding-weights --model "${1}"  --log_level DEBUG
