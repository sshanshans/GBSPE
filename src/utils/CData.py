import math
import pickle
import numpy as np
from collections import defaultdict
from src._helpers.math import *
from src._helpers.random import *
from src._helpers.check import *
from src.utils.SubTuple import SubTuple
from src.utils.CovMat import CovMat
import pprint
import itertools
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class CData(defaultdict):
    """
    Class to encode coefficient data and hafnain data
    """
    def __init__(self, N, K, B, phi, cdict):
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
        self.coeff_data = cdict
        self.haf_data = {}
        self._check_phi()

    def _check_phi(self):
        if self.phi not in {'haf', 'hafsq', 'noise'}:
            raise ValueError("Invalid phi value. Must be 'haf', 'hafsq', or 'noise'.")
            
    def __reduce__(self):
        # Return a tuple with the class, the arguments to pass to __init__, and the instance's state
        return (self.__class__, (self.N, self.K, self.B, self.phi, self.coeff_data), self.__dict__)
    
    def __setstate__(self, state):
        # Update the instance's __dict__ with the unpickled state
        self.__dict__.update(state)

    def save(self, filename):
        # Save the current object to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
          return pickle.load(f)

    def update_coeffs(self, num_cores=8):
        '''
        For each I, assign the weights to be a random number between [-1, 1] with uniform distribution.
        Parallelize the update process.
        '''
        if num_cores > 1:
            # Use a ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = {
                    executor.submit(self._assign_random_coeff, I): I
                    for I in self.coeff_data.keys()
                }
    
                # Wait for all futures to complete
                for future in as_completed(futures):
                    I = futures[future]
                    try:
                        future.result()  # Block until each task is done
                    except Exception as e:
                        print(f"Error updating coeff for {I}: {e}")
        else:
            for I in self.coeff_data.keys():
                self._assign_random_coeff(I)

    def _assign_random_coeff(self, I, real_bool=False):
        """
        Helper function to assign a random coefficient to I.
        """
        if real_bool == True:
            self.coeff_data[I] = generate_rand_coeff()
        else:
            a = generate_rand_coeff()
            b = generate_rand_coeff()
            self.coeff_data[I] = complex(a, b)
            

    ## Initialize hafnian dictionary
    def compute_haf_for_pair(self, new_tuple):
        if new_tuple not in self.haf_data:
            I = SubTuple(new_tuple, self.B, self.phi)
            I.compute_phival()
            self.haf_data[new_tuple] = I.phival  # or perform computation here
            return new_tuple  # Return the computed key

    def populate_haf_data_parallel(self, num_cores=8):
        new_tuples_to_process = []
        for i, I in enumerate(self.coeff_data.keys()):
                for J in list(self.coeff_data.keys())[i:]:  # Avoid redundant pairs
                    new_tuple = tuple(map(lambda x, y: x + y, I, J))
                    new_tuples_to_process.append(new_tuple)
        unique_tuples_to_process = list(set(new_tuples_to_process))  # This ensures only unique tuples are processed
        
        if num_cores > 1:
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                futures = {executor.submit(self.compute_haf_for_pair, new_tuple): new_tuple for new_tuple in unique_tuples_to_process}
                for future in futures:
                    try:
                        result = future.result()  # Blocks until the future is done
                        #if result is not None:
                        #    print(f"Computed haf_data for: {result}")  # Handle the result
                    except Exception as e:
                        print(f"Error computing pair {futures[future]}: {e}")
        else:
            for new_tuple in unique_tuples_to_process:
                self.compute_haf_for_pair(new_tuple)

    ## compute various quantities
    @property
    def gt(self):
        gt_sum = 0
        for I, aI in self.coeff_data.items():
            gt_sum += aI * self.haf_data[I]
        return gt_sum

    @property
    def Q_GBS(self, real_bool=False):
        # only valid for hafsq
        qgbs_sum = 0
        for I, aI in self.coeff_data.items():
            if real_bool == True:
                qgbs_sum += aI**2 * ifac(I) * self.haf_data[I]
            else:
                aIsq = aI * aI.conjugate()
                qgbs_sum += aIsq.real * ifac(I) * self.haf_data[I]
        return 1/self.B.d * qgbs_sum

    @property
    def Q_MC(self, real_bool=False):
        hafsum = 0
        if real_bool == True:
            for i, I in enumerate(self.coeff_data.keys()):
                for J in list(self.coeff_data.keys())[i:]:
                    L = tuple(map(lambda x, y: x + y, I, J))
                    if I == J:
                        aI = self.coeff_data[I]
                        hafsum += aI**2 * self.haf_data[L]
                    else:
                        aI = self.coeff_data[I]
                        aJ = self.coeff_data[J]
                        hafsum += 2 * aI * aJ * self.haf_data[L]

        else:
            for i, I in enumerate(self.coeff_data.keys()):
                for J in list(self.coeff_data.keys())[i:]:
                    L = tuple(map(lambda x, y: x + y, I, J))
                    if I == J: 
                        aI = self.coeff_data[I]
                        aIsq = aI * aI.conjugate()
                        hafsum += aIsq.real * self.haf_data[L]
                    else:
                        aI = self.coeff_data[I]
                        aJ = self.coeff_data[J]
                        aIaJ = aI * aJ.conjugate() + aI.conjugate() * aJ
                        hafsum += aIaJ.real * self.haf_data[L]
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
            start_time = time.time()
            mu = self.gt
            end_time = time.time()
            t1 = end_time - start_time
            log_message(f'gt: {mu}', f)
            start_time = time.time()
            qmc = self.Q_MC
            end_time = time.time()
            t2 = end_time - start_time
            log_message(f'mc: {qmc}', f)
            start_time = time.time()
            qgbs = self.Q_GBS
            end_time = time.time()
            t3 = end_time - start_time
            log_message(f'gbs: {qgbs}', f)
            if qgbs <= qmc:
                flag_gbs_more_efficient = 1
            else:
                flag_gbs_more_efficient = 0
            log_message(f'gbs_eff: {flag_gbs_more_efficient}', f)
            log_message(f'time for gt:{t1}', f)
            log_message(f'time for mc:{t2}', f)
            log_message(f'time for gbs:{t3}', f)
        return flag_gbs_more_efficient

    def enumerate_all_pI(self):
        keys, weights = [], []
        for I in self.coeff_data.keys():
            keys.append(I)
            if self.phi == 'haf':
                hafval = self.haf_data[I] ** 2
            elif self.phi == 'hafsq':
                hafval = self.haf_data[I]
            else:
                raise ValueError("Invalid phi value. Must be 'haf' or 'hafsq'.")
            pI = self.B.d * (1 / ifac(I)) * hafval
            pI = np.real(pI)
            weights.append(pI)
        return keys, weights
            

