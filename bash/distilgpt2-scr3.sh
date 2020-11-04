#!/bin/bash
#
#SBATCH --job-name=run_scrambled_distilgpt_scr3
#SBATCH --output=run_scrambled_distilgpt2_scr3_%j.out
#SBATCH --error=run_scrambled_distilgpt2_scr3_%j.err
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.
#SBATCH -t 15:00:00
timestamp() {
  date +"%T"
}
echo 'Executing run_scrambled_distilgpt2_scr3'
timestamp
filename="run_scrambled_distilgpt2_scr3_$(date '+%Y%m%d%T').txt"
cd /om/user/gretatu/neural-nlp
source activate brainmodeling

export MODELNAME=distilgpt2
python neural_nlp run --benchmark Pereira2018-encoding-scrambled3 --model ${MODELNAME}  --log_level DEBUG > $filename
echo 'Finished!'
timestamp
