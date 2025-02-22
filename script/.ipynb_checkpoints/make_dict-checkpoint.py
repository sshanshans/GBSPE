from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import itertools
from itertools import product
import time
import pickle
import argparse
from src._helpers.check import *

def generate_tuples_sum_2K(N, K):
    completed_tasks = 0
    max_value = 2 * K
    cdict = {}
    for I in generate_combinations(N, K, max_value):
        completed_tasks += 1
        cdict[I] = None
    print(f"{completed_tasks} tasks completed")
    return cdict

def make_hdict(cdict):
    cdict_keys = list(cdict.keys())
    hset = {
        tuple(map(sum, zip(I, J)))
        for idx, I in enumerate(cdict_keys)
        for J in cdict_keys[idx:]  # Avoid redundant pairs
    }
    # Also include keys from cdict in the hdict
    hset.update(cdict_keys)
    return {item: None for item in hset}

def generate_combinations(N, K, max_value):
    """
    Generate all combinations of length N where the elements sum to 2*K
    and each element is in the range [0, max_value].
    """
    def backtrack(current_tuple, remaining_sum, remaining_elements):
        # Base case: If we've filled N elements and sum is exactly 2*K
        if remaining_elements == 0:
            if remaining_sum == 0:
                yield tuple(current_tuple)
            return
        
        # Try adding numbers to the current tuple
        for i in range(min(remaining_sum, max_value) + 1):
            current_tuple.append(i)
            yield from backtrack(current_tuple, remaining_sum - i, remaining_elements - 1)
            current_tuple.pop()

    return backtrack([], 2 * K, N)

def main():
    parser = argparse.ArgumentParser(description="Run experiment with specified parameters.")
    parser.add_argument('-N', type=int, required=True, help='Number of trials (N)')
    parser.add_argument('-K', type=int, required=True, help='Number of successes (K)')
    args = parser.parse_args()

    base_folder = '/work/GBSPE/exp/dict/'
    N = args.N
    K = args.K
    filename1 = f'{base_folder}cdata/N-{N}-K-{K}.pkl'
    filename2 = f'{base_folder}hdata/N-{N}-K-{K}.pkl'
    start_time = time.time()
    cdict = generate_tuples_sum_2K(N, K)
    save_dict(cdict, filename1)
    hdict = make_hdict(cdict)
    save_dict(hdict, filename2)
    end_time = time.time()
    print('time to generate all tuples:', end_time - start_time)

if __name__ == "__main__":
    main()