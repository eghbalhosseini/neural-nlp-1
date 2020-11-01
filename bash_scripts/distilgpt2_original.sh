#!/bin/bash
#
#SBATCH --job-name=run_scrambled_distilgpt2_scrambled_original
#SBATCH --output=run_scrambled_distilgpt2_scrambled_original_%j.out
#SBATCH --error=run_scrambled_distilgpt2_scrambled_original_%j.err
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.
#SBATCH -t 06:00:00


timestamp() {
  date +"%T"
}

echo "Executing run_scrambled_distilgpt2_scrambled_original"
timestamp

filename="run_scrambled_distilgpt2_scrambled_original_$(date '+%Y%m%d%T').txt"

cd /om/user/ckauf/neural-nlp

export MODELNAME=distilgpt2

python neural_nlp run --benchmark Pereira2018-encoding-scrambled-original --model ${MODELNAME}  --log_level DEBUG > $filename

echo "Finished!"
timestamp
