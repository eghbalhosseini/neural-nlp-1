#!/bin/sh
models="gpt2 bert-large-uncased-whole-word-masking"
versions="Scr1 Scr3 Scr5 Scr7 lowPMI random backward"
embeddings="last-tok avg-tok"
periods="True False"

for model in $models ; do
    for version in $versions ; do
      for embedding in $embeddings ; do
        for period in periods ; do
          sbatch get-actv.sh $model $version $embedding $period
          done
        done
    done
done