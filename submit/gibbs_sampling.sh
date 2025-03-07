#!/bin/bash
python data_generation.py --datafile data/data_experiment
for i in $(seq 1 5);
do
    python gibbs_sampler.py --n_clusters $i --savepath data/gibbs_${i}.npy --n_iter 2000 --data_path data/data_experiment.npy
done
