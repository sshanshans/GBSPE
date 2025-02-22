import pickle
import os
import numpy as np
from src.methods.ProcessingMethod import ProcessingMethod
from src._helpers.math import *
from src._helpers.gensam import *
import random

class GBSP(ProcessingMethod):
    def __init__(self):
        self.name = 'GBSP'
        
    def process(self, T, num_samples):
        es = []
        count_dict = create_default_nested_dict(T.data, 0)
        count_a0 = 0
        xs = generate_sample(T, num_samples)
        d_inv = T.B.dinv
        for n, x in enumerate(xs):
            es.append(self.evaluate_sample(T, x))
            k = sum(x)
            if ~np.isnan(k):
                if k == 0:
                    count_a0 +=1
                else:
                    count_dict[k][x] += 1
            # Compute the integral by summing coeff and the probability estimation (remember to divide by num_samples)
            current_estimate = np.sum([aI * np.sqrt(d_inv * ifac(I) * query_single_I_from_count_dict(count_dict, I)/n) for I, aI in T.enumerate_all_aI()]) + T.a0 * np.sqrt(d_inv * count_a0/n)
            es.append(current_estimate)
        return es

def query_single_I_from_count_dict(count_dict, I):
    # note that the enumerate_all_aI does not contain a0
    k = sum(I)
    return count_dict[k][I]

def create_default_nested_dict(d, default_value):
    if isinstance(d, dict):
        return {key: create_default_nested_dict(value, default_value) for key, value in d.items()}
    else:
        return default_value