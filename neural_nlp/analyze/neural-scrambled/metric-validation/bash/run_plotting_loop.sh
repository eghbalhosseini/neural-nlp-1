#!/bin/bash
#
#SBATCH --job-name=plot_val
#SBATCH --output=plot_val%j.out
#SBATCH --error=plot_val_%j.err
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH -t 00:05:00

timestamp() {
  date +"%T"
}

echo 'Plotting'
timestamp

source activate brainmodeling

cd /om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/


filename="${1}_${4}_${3}_finalPeriod=${2}_${5}.txt"
    
python plotting_script.py --model_identifier "${1}" --final_period "${2}" --sentence_embedding "${3}" --metric "${4}" --norm_method "${5}" > "/om/user/ckauf/neural-nlp/neural_nlp/analyze/neural-scrambled/metric-validation/bash/$filename"

timestamp

echo 'Done!'