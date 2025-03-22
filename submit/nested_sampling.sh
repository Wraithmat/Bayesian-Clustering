#!/bin/bash
for j in 70 100;
do
for i in $(seq 1 5);
do
    python nested_sampling.py --n_clusters $i --outputfile 1003_${i}_seed_1_nlive_${j}.txt --steps 1500 --seed 1 --N_live $j --data ./data_new.npy
done
done