#!/bin/bash
#
#SBATCH --job-name=test-actv
#SBATCH --output=test-actv_%j.out
#SBATCH --error=test-actv_%j.err
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH -t 02:00:00

timestamp() {
  date +"%T"
}

echo 'Executing run_scrambled'
timestamp

filename="test-actv_$(date '+%Y%m%d%T').txt"

source /om2/user/gretatu/anaconda/etc/profile.d/conda.sh
conda activate control-neural


cd /om/user/`whoami`/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/

python get_activations.py --model "${1}" --scrambled_version "${2}" --sentence_embedding "${3}" --final_period "${4}" > "/om/user/`whoami`/neural-nlp/bash/$filename"

timestamp

echo 'All complete!'

# RUN LIKE THIS (in shell):
# for cond in scrambled-original scrambled1 scrambled3 scrambled5 scrambled7 scrambled-lowpmi scrambled-random scrambled-backward; do sbatch loop_scr_all.sh $cond; done
