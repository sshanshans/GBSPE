#!/bin/bash

for N in {3..6}; do
    for K in {2..8}; do
        nohup python -u make_dict.py -N $N -K $K > ../exp/exp3/logs/gendict/N_$N-K_$K.txt 2>&1 &
    done
done