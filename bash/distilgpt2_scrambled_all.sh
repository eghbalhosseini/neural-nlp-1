#!/bin/bash
#
#SBATCH --job-name=run_scrambled
#SBATCH --output=run_scrambled_%j.out
#SBATCH --error=run_scrambled_%j.err
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.
#SBATCH -t 24:00:00

timestamp() {
  date +"%T"
}

echo 'Executing run_scrambled'
timestamp

filename="run_scrambled_$(date '+%Y%m%d%T').txt"

cd /om/user/gretatu/neural-nlp
source activate brainmodeling

export MODELNAME=distilgpt2

#for i in scrambled-original scrambled1 scrambled3 scrambled5 scrambled7 scrambled-lowpmi

for i in scrambled3 scrambled5 scrambled7

do
    python neural_nlp run --benchmark Pereira2018-encoding-${i} --model ${MODELNAME}  --log_level DEBUG > "/om/user/gretatu/neural-nlp/bash/$filename"

    echo 'Finished benchmark!'
    timestamp

done
echo 'All complete!'
