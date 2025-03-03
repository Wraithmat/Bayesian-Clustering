#!/bin/bash
python data_generation.py
for i in $(seq 1 5);
do
    python gibbs_sampler.py --n_clusters $i --savepath data/gibbs_${i}.npy --n_iter 1500
done
