import math
import pickle
import numpy as np
from collections import defaultdict
from src._helpers.math import *
from src._helpers.random import *
from src._helpers.check import *
from src.utils.SingleTuple import SingleTuple
from src.utils.CovMat import CovMat
import pprint
import itertools
import random

class CoeffDict(defaultdict):
    """
    Class of a tuple list
    """
    def __init__(self, N, K, B, phi):
        # Check if N and K are positive integers
        if not (isinstance(N, int) and N > 0):
            raise ValueError("N must be a positive integer")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer")
        
        # Check if B is an instance of CovMat
        if not isinstance(B, CovMat):
            raise ValueError("B must be an instance of CovMat")
        
        # Check if phi is either 'haf' or 'hafsq' or noise
        if phi not in ['haf', 'hafsq', 'noise']:
            raise ValueError("phi must be either 'haf' or 'hafsq' or 'noise'")
        
        self.N = N     # dimension of the integral
        self.K = K     # order of multivariate polynomial
        self.B = B     # single block covariance matrix in GE
        self.phi = phi # choice of 'haf' or 'hafsq' or 'noise'
        self.data = self._initialize_dict()
        self.haf_data = self._initialize_haf_dict()
        self._check_phi()

    def __reduce__(self):
        # Return a tuple with the class, the arguments to pass to __init__, and the instance's state
        return (self.__class__, (self.N, self.K, self.B, self.phi), self.__dict__)
    
    def __setstate__(self, state):
        # Update the instance's __dict__ with the unpickled state
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
          return pickle.load(f)

    @property
    def coeffnum(self):
        num = 0
        for key in list(self.data.keys()):
            num_to_add = int(len(self.data[key]))
            num = num_to_add + num
        return num

    def _check_phi(self):
        if self.phi not in {'haf', 'hafsq', 'noise'}:
            raise ValueError("Invalid phi value. Must be 'haf', 'hafsq', or 'noise'.")

    ## initialize coefficient dictionary
    def _initialize_dict(self):
        """
        Initialize the data dictionary using the
        following hierarchical structure
        # Level 1: Initialize cells 0, 2, ..., 2K
        # Level 2: tuples of sum 2*i 
        # Level 3: SingleTuple class
        Populate all entries and then assign uniform random aI
        """
        data_dict = {2 * i: {} for i in range(0, self.K + 1)}
        return data_dict

    def populate(self):
        completed_tasks = 0
        total_tasks = (2 * self.K + 1) ** self.N
        #print('total', total_tasks)
        for I in itertools.product(range(2 * self.K + 1), repeat=self.N):
            self._process_single_I_populate(I)
            completed_tasks += 1
            #print(f"{completed_tasks} tasks completed", end="\r")

    def _process_single_I_populate(self, I):
        k = sum(I)
        if k % 2 == 0 and k <= 2 * self.K:
            if I not in self.data[k]:
                self.data[k][I] = SingleTuple(I, self.B, self.phi)

    def update_coeffs(self):
        for key in self.data.keys():
            for I in self.data[key].keys():
                self._process_single_coeff(key, I)

    def  _process_single_coeff(self, key, I):
        '''
        For each I, assign the weights to be a random number between [-1, 1] with uniform distribution
        '''
        if self.phi == 'haf':
            self.data[key][I].aI = generate_rand_coeff()
        else:
            self.data[key][I].aI = generate_rand_coeff()

    def enumerate_all_aI(self):
        for key in self.data.keys():
            for I in self.data[key].keys():
                yield (I, self.data[key][I].aI)

    ## initialize hafnian dictionary
    def _initialize_haf_dict(self):
        """
        Initialize the data dictionary using the
        following hierarchical structure
        # Level 1: Initialize cells 0, 2, ..., 2K
        # Level 2: tuples of sum 2*i 
        # Level 3: SingleTuple class
        """
        data_dict = {2 * i: {} for i in range(2*self.K + 1)}
        return data_dict

    def populate_haf_dict(self):
        completed_tasks = 0
        for (I, aI) in self.enumerate_all_aI():
            for (J, aJ) in self.enumerate_all_aI():
                L = tuple(map(lambda x, y: x + y, I, J))
                self._process_single_I_populate_haf(L)
                completed_tasks += 1
                #print(f"{completed_tasks} tasks completed", end="\r")

    def _process_single_I_populate_haf(self, I):
        k = sum(I)
        if I not in self.haf_data[k]:
            self.haf_data[k][I] = SingleTuple(I, self.B, self.phi)
    
    def update_all(self, propertyname):
        completed_tasks = 0
        for key in self.haf_data.keys():
            for I in self.haf_data[key].keys():
                self._update_single_I(I, propertyname)
                completed_tasks += 1
                #print(f"{completed_tasks} tasks completed", end="\r")

    def _update_single_I(self, I, propertyname):
        k = sum(I)
        if propertyname == 'phival':
            if self.haf_data[k][I].phival is None:
                self.haf_data[k][I].compute_phival()
        else:
            raise ValueError("Invalid propertyname: {}".format(propertyname))

    def enumerate_all_haf(self):
        for key in self.haf_data.keys():
            for I in self.haf_data[key].keys():
                yield (I, self.haf_data[key][I].phival)

    ## compute various quantities
    @property
    def gt(self):
        gt_sum = 0
        for (I, aI) in self.enumerate_all_aI():
            k = sum(I)
            vv = self.haf_data[k][I].phival
            gt_sum = gt_sum + aI * vv
        return gt_sum

    @property
    def Q_GBS(self):
        if self.phi == 'haf':
            qgbs_sum = 0
            for (I, aI) in self.enumerate_all_aI():
                k = sum(I)
                tt = self.haf_data[k][I]
                vv = tt.phival
                qgbs_sum = qgbs_sum + aI * tt.ifac / vv
            return np.abs(1/self.B.d * qgbs_sum * self.gt)
        else:
            qgbs_sum = 0
            for (I, aI) in self.enumerate_all_aI():
                k = sum(I)
                tt = self.haf_data[k][I]
                vv = tt.phival
                qgbs_sum = qgbs_sum + aI**2 * tt.ifac * vv
            return 1/self.B.d * qgbs_sum

    @property
    def Q_MC(self):
        hafsum = 0 
        for (I, aI) in self.enumerate_all_aI():
            for (J, aJ) in self.enumerate_all_aI():
                L = tuple(map(lambda x, y: x + y, I, J))
                k = sum(L)
                vv = self.haf_data[k][L].phival
                hafsum = hafsum + aI * aJ * vv
        return  hafsum 
    
    def print_log(self, path_to_log):
        with open(path_to_log, 'w') as f:
            # Redirect print statements to the text file
            log_message('Logging started...', f)
            log_message('========================', f)
            log_message('Basic information...', f)
            log_message(f'N: {self.N}', f)
            log_message('Bmat: ', f)
            log_message(f'{self.B.bmat}', f)
            log_message(f'd: {self.B.d}', f)
            log_message(f'dinv: {self.B.dinv}', f)
            log_message(f'K: {self.K}', f)
            log_message(f'phi: {self.phi}', f)
            mu = self.gt
            log_message(f'gt: {mu}', f)
            qmc = self.Q_MC
            log_message(f'mc: {qmc}', f)
            qgbs = self.Q_GBS
            log_message(f'gbs: {qgbs}', f)
            if qgbs <= qmc:
                flag_gbs_more_efficient = 1
            else:
                flag_gbs_more_efficient = 0
            log_message(f'gbs_eff: {flag_gbs_more_efficient}', f)
        return flag_gbs_more_efficient

