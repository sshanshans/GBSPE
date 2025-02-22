#!/bin/bash

for N in {3..4}; do
    for K in {2..3}; do
        nohup python -u run-percentage-haf.py -N $N -K $K > ../exp/logs/sample-haf/N_$N-K_$K.txt 2>&1 &
        wait
        nohup python -u run-percentage-hafsq.py -N $N -K $K > ../exp/logs/sample-hafsq/N_$N-K_$K.txt 2>&1 &
        wait
    done
done