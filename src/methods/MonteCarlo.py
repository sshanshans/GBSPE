import pickle
import os
import numpy as np
from src.methods.ProcessingMethod import ProcessingMethod
from src._helpers.math import *

class MonteCarlo(ProcessingMethod):
    def __init__(self):
        self.name = 'MC'
        
    def process(self, T, num_samples):
        es = []
        xs = self.generate_sample(T, num_samples)
        for x in xs:
            es.append(self.evaluate_sample(T, x))
        return es

    def generate_sample(self, T, num_samples):
        seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
        np.random.seed(seed)
        if T.phi == 'haf':
            mean = np.zeros(T.N)
            cov = T.B.bmat
        else:
            mean = np.zeros(2*T.N)
            cov = T.B.convert_bmat_to_cov_normal()
        return np.random.multivariate_normal(mean, cov, num_samples)

    def evaluate_sample(self, T, x):
        if T.phi == 'haf':
            f = product_of_powers_single
        else:
            f = product_of_powers_double
        return T.a0 + np.sum([aI * f(x, I) for I, aI in T.coeff_data.items()])

class MonteCarloS(ProcessingMethod):
    # only for T.phi = haf
    def __init__(self):
        self.name = 'MCS'
        
    def process(self, T, num_samples):
        xs = self.generate_sample(T, num_samples)
        es = self.evaluate_sample(T, xs)
        return es

    def generate_sample(self, T, num_samples):
        seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
        np.random.seed(seed)
        I_list = list(T.coeff_data.keys())
        samples = np.random.choice(len(I_list), size=num_samples)
        return [I_list[s] for s in samples]

    def evaluate_sample(self, T, sampled_items):
        es = []
        val = 0
        observed_tuples = set()  # Initialize an empty set
        
        for I in sampled_items:
            if I not in observed_tuples:
                val += T.haf_data[I] * T.coeff_data[I]  # Increment val
                observed_tuples.add(I)  # Add item to the set
            es.append(val)
        return es
