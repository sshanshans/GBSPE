import numpy as np
import itertools
import math
import random
from pathlib import Path
from collections import defaultdict
from src.utils.PData import PData
from src._helpers.math import *
from src._helpers.random import * 
from src._helpers.check import * 
import pprint
import argparse
import string
import os
import concurrent.futures
import pickle
import time
import matplotlib.pyplot as plt
import mmap

# Process each `jj` value in parallel
def _process_jj(arg):
    T, jj, path_to_folder, r = arg
    T.update_a(jj)
    gt = T.compute_gt()
    qmc = T.compute_qmc()
    vmc = qmc - gt**2
    qgbs = T.compute_qgbs()
    vgbs = 1/4 * (qgbs - gt**2)
    path_to_log = f"{path_to_folder}/log{r}_{jj}.txt"
    T.print_log2(path_to_log, gt, vmc, vgbs)
    return jj

def main():
    parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
    parser.add_argument('-N', type=int, required=True, help='Number of dimension (N)')
    parser.add_argument('-K', type=int, required=True, help='Number of taylor series coeff (K)')
    parser.add_argument('-S', type=int, default=0, help='Starting point (default: 0)')
    parser.add_argument('-E', type=int, default=30, help='Ending point (default: 30)')
    args = parser.parse_args()

    base_folder = '/work/GBSPE/exp/haf'
    dict_folder = '/work/GBSPE/exp/dict'
    phi = 'haf'
    N = args.N
    K = args.K
    start_num = args.S
    max_num = args.E
    exp_id = f'{phi}-N_{N}-K_{K}'
    path_to_folder = f'{base_folder}/{exp_id}'
    check_and_create_folder(path_to_folder)
    
    #print(os.sched_getaffinity(0))

    cdict_filename = f'{dict_folder}/cdata/N-{N}-K-{K}.pkl'
    hdict_filename = f'{dict_folder}/hdata/N-{N}-K-{K}.pkl'
    cdict = load_dict_from_file(cdict_filename)
    hdict = load_dict_from_file(hdict_filename)
    haf_data_folder = f'{base_folder}/haf_data/N-{N}-K-{K}/'
    check_and_create_folder(haf_data_folder)
    
    T =  PData(N, K, phi, cdict, hdict)

    for r in range(start_num, max_num):
        print('Initialized #', r)
        T.update_B(haf_data_folder, r)

        # Prepare arguments for each run (each jj iteration)
        args_list = [(T, jj, path_to_folder, r) for jj in range(200)]

        # Determine number of workers
        num_workers = os.cpu_count()

        # Use ProcessPoolExecutor for parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks in parallel and collect the results
            futures = [executor.submit(_process_jj, arg) for arg in args_list]
            
            # Optionally, gather the results as they complete
            for future in concurrent.futures.as_completed(futures):
                jj = future.result()  # This can raise an exception if any occurred during execution
                print(f'Process r_{r}: jj_{jj} completed')

        print(f'Process N_{N}-K_{K}-r_{r} completed')

if __name__ == "__main__":
    main()
