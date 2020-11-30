#!/bin/bash
#
#SBATCH --job-name=skipthoughts-all
#SBATCH --output=skipthoughts-all_%j.out
#SBATCH --error=skipthoughts-all_%j.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 08:00:00

timestamp() {
  date +"%T"
}

echo 'Executing skipthoughts-all'
timestamp

filename="skipthoughts-all_$(date '+%Y%m%d%T').txt"

cd /om/user/ckauf/neural-nlp
source activate brainmodeling

export MODELNAME=skip-thoughts

python neural_nlp run --benchmark Pereira2018-encoding-"${1}" --model ${MODELNAME}  --log_level DEBUG > $filename

timestamp

echo 'All complete!'