#!/bin/bash
#
#SBATCH --job-name=run_all_scrambled
#SBATCH --output=run_all_scrambled_%j.out
#SBATCH --error=run_all_scrambled_%j.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 28:00:00

timestamp() {
  date +"%T"
}

echo 'Executing run_scrambled'
timestamp

filename="run_all_scrambled_$(date '+%Y%m%d%T').txt"

cd /om/user/`whoami`/neural-nlp
source activate brainmodeling

export MODELNAME=xlnet-large-cased

#for i in scrambled-original scrambled1 scrambled3 scrambled5 scrambled7 scrambled-lowpmi

python neural_nlp run --benchmark Pereira2018-encoding-"${1}" --model ${MODELNAME}  --log_level DEBUG > "/om/user/`whoami`/neural-nlp/bash/$filename"

timestamp

echo 'All complete!'

# RUN LIKE THIS (in shell):
# for cond in scrambled-original scrambled1 scrambled3 scrambled5 scrambled7 scrambled-lowpmi scrambled-random scrambled-backward; do sbatch loop_scr_all.sh $cond; done
