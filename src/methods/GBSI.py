import pickle
import os
import numpy as np
from src.methods.ProcessingMethod import ProcessingMethod
from src._helpers.math import *
from src._helpers.gensam import *
import random

class GBSI(ProcessingMethod):
    def __init__(self):
        self.name = 'GBSI'
    
    def process(self, T, num_samples):
        es = []
        xs = generate_sample(T, num_samples)
        for x in xs:
            es.append(evaluate_sample(T, x))
        return es

class GBSIn(ProcessingMethod):
    '''
    This is the class of GBS estimators that uses existing samples
    '''
    def __init__(self):
        self.name = 'GBSIn'
    
    def process(self, T, xs):
        es = []
        total = len(xs)
        count = 0
        for x in xs:
            count = count + 1
            print(f"{count}/{total}", end="\r")
            x = tuple(x.flatten())
            es.append(evaluate_sample(T, x))
        return es

class GBSIs(ProcessingMethod):
    '''
    This is the class of GBS estimators that uses existing samples with specially assigned aI's 
    '''
    def __init__(self):
        self.name = 'GBSIs'
    
    def process(self, T, xs, r):
        es = []
        total = len(xs)
        count = 0
        for x in xs:
            count = count + 1
            print(f"{count}/{total}", end="\r")
            x = tuple(x.flatten())
            es.append(evaluate_sample_special(T, x, r))
        return es

'''
Some utilities functions
'''
def evaluate_sample(T, x):
    k = sum(x)
    if np.isnan(k):
        return 0
    if k == 0:
        return T.B.dinv * T.a0
    elif k not in T.data or x not in T.data[k]:
        return 0
    else:
        I = T.data[k][x]
        return T.B.dinv * I.ifac * I.aI

def evaluate_sample_special(T, x, r):
    k = sum(x)
    ifactorial = ifac(x)
    if k == 0:
        return T.B.dinv * T.a0
    else:
        return T.B.dinv * (r**k)






