#!/bin/bash
#
#SBATCH --job-name=skipthoughts-scr1
#SBATCH --output=skipthoughts-scr1%j.out
#SBATCH --error=skipthoughts-scr1%j.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 08:00:00
timestamp() {
  date +"%T"
}
echo 'Executing skipthoughts-scr1'
timestamp
filename="skipthoughts-scr1_$(date '+%Y%m%d%T').txt"
cd /om/user/ckauf/neural-nlp
source activate brainmodeling

export MODELNAME=skip-thoughts
python neural_nlp run --benchmark Pereira2018-encoding-scrambled1 --model ${MODELNAME} --log_level DEBUG > $filename
echo 'Finished!'
timestamp