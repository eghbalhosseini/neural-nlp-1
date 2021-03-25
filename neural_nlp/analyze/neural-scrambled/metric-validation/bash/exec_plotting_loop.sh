#!/bin/sh
models="gpt2 bert-base-uncased"
periods="True False"
embeddings="last-tok avg-tok"
metrics="pearson spearman cosine euclidean"
norm_methods="no_norm clip row_normalize"

for model_identifier in $models ; do
    for final_period in $periods ; do
        for sentence_embedding in $embeddings ; do
            for metric in $metrics ; do
                for norm_method in $norm_methods ; do
                    sbatch run_plotting_loop.sh $model_identifier $final_period $sentence_embedding $metric $norm_method
                done
            done
        done
    done
done