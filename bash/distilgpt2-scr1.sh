#!/bin/bash
#
#SBATCH --job-name=run_scrambled_gpt2_scr1
#SBATCH --output=run_scrambled_gpt2_scr1_%j.out
#SBATCH --error=run_scrambled_gpt2_scr1_%j.err
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.
#SBATCH -t 06:00:00
timestamp() {
  date +"%T"
}
echo 'Executing run_scrambled_distilgpt2_scr1'
timestamp
filename="run_scrambled_distilgpt2_scr1_$(date '+%Y%m%d%T').txt"
cd /om/user/gretatu/neural-nlp
source activate brainmodeling

export MODELNAME=distilgpt2
python neural_nlp run --benchmark Pereira2018-encoding-scrambled1 --model ${MODELNAME}  --log_level DEBUG > $filename
echo 'Finished!'
timestamp
