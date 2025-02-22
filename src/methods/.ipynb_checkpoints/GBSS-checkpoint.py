import pickle
import os
import numpy as np
from src.methods.ProcessingMethod import ProcessingMethod
from src._helpers.math import *
from src._helpers.gensam import *
import random

class GBSS(ProcessingMethod):
    def __init__(self):
        self.name = 'GBSS'
        
    def process(self, T, num_samples):
        xs = generate_sample(T, num_samples)
        es = self.evaluate_sample(T,xs)
        return es

    def evaluate_sample(self, T, sampled_items):
        es = []
        val = 0
        observed_tuples = set()  # Initialize an empty set
        
        for I in sampled_items:
            if I not in observed_tuples and not np.isnan(I).any():
                val += T.haf_data[I] * T.coeff_data[I]  # Increment val
                observed_tuples.add(I)  # Add item to the set
            es.append(val)
        return es