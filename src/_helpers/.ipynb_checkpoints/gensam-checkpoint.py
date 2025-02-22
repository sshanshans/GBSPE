import os
import numpy as np

def generate_sample(T, num_samples):
    seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
    np.random.seed(seed)
    keys, weights = T.enumerate_all_pI()
    nan_tuple = tuple([np.nan] * T.N)
    keys.append(nan_tuple)
    weights.append(1 - sum(weights))
    if len(keys) != len(weights):
        raise ValueError("The length of keys and weights must be the same")
    samples = np.random.choice(len(keys), size=num_samples, p=weights)
    return [keys[s] for s in samples]