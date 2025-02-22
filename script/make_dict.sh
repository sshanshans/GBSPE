#!/bin/bash

for N in {3..4}; do
    for K in {2..3}; do
        nohup python -u make_dict.py -N $N -K $K > ../exp/logs/gendict/N_$N-K_$K.txt 2>&1 &
    done
done