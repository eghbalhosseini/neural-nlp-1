#!/bin/bash
#
#SBATCH --job-name=get-actv
#SBATCH --output=get-actv_%j.out
#SBATCH --error=get-actv_%j.err
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH -t 00:30:00

timestamp() {
  date +"%T"
}

echo 'Fetching activations'
timestamp

filename="get-actv_$(date '+%Y%m%d%T').txt"

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate control-neural

cd /om/user/`whoami`/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/

python get_activations.py --model "${1}" --scrambled_version "${2}" --sentence_embedding "${3}" --final_period "${4}" > "/om/user/`whoami`/neural-nlp/bash/$filename"

timestamp

echo 'All complete!'