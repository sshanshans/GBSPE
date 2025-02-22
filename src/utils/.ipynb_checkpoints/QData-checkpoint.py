import math
import pickle
import numpy as np
from collections import defaultdict
from src._helpers.math import *
from src._helpers.random import *
from src._helpers.check import *
from src._helpers.optimize import *
from src.utils.SubTuple import SubTuple
from src.utils.CovMat import CovMat
import pprint
import itertools
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

class QData(defaultdict):
    """
    Class to encode coefficient data and hafnain data
    """
    def __init__(self, N, K, phi, cdict, hdict):
        # Check if N and K are positive integers
        if not (isinstance(N, int) and N > 0):
            raise ValueError("N must be a positive integer")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer")
            
        # Check if phi is either'hafsq' or noise
        if phi not in ['haf']:
            raise ValueError("phi must be set to be haf.")
        
        self.N = N     # dimension of the integral
        self.K = K     # order of multivariate polynomial
        self.phi = phi # choice of 'hafsq' or 'noise'
        self.coeff_data = cdict
        self.haf_data = hdict
        self.B = None
        self.vdet = None
        
    def __reduce__(self):
        # Return a tuple with the class, the arguments to pass to __init__, and the instance's state
        return (self.__class__, (self.N, self.K, self.phi, self.coeff_data), self.__dict__)
    
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

    def update_coeffs(self, rseed=532):
        '''
        For each I, assign the weights to on the hypersphere with uniform distribution.
        '''
        keys = list(self.coeff_data.keys())
        num_points = 1
        dimension = len(keys)
        #vector = sample_points_uniform(num_points, dimension, rseed).flatten()
        vector = sample_points_on_sphere_real(num_points, dimension, rseed).flatten()
        self.coeff_data = {key: val for key, val in zip(keys, vector)}

    def update_coeffs_positive(self, rseed=532):
        '''
        For each I, assign the weights to on the hypersphere with uniform distribution.
        '''
        keys = list(self.coeff_data.keys())
        num_points = 1
        dimension = len(keys)
        #vector = sample_points_uniform(num_points, dimension, rseed).flatten()
        vector = sample_points_on_sphere_real(num_points, dimension, rseed).flatten()
        self.coeff_data = {key: np.abs(val) for key, val in zip(keys, vector)}

    def update_B(self, rseed=532):
        Bmat = generate_random_Bmat(self.N, rseed)
        lambdas = np.linalg.eigvals(Bmat)
        self.vdet = vandermonde_determinant(lambdas)
        B = CovMat(Bmat)
        new_B, t = B.compute_klevel_compatible_bmat(self.K)
        self.B = new_B
        bmat = self.B.bmat
        start_time = time.time()
        for I in self.haf_data.keys():
            val = haf_I(bmat, I)
            self.haf_data[I] = np.real(val) # consider only real B
        end_time = time.time()
        print(f'total time to compute haf data table: {end_time - start_time}')
        t_factor = t**(-self.K)
        for I in self.coeff_data.keys():
            aI = self.coeff_data[I]
            self.coeff_data[I] = aI * t_factor

    def extract_short_cut(self, I):
        d = self.B.d
        dsqrt = np.sqrt(d)
        aI = self.coeff_data[I]
        val_I = self.haf_data[I]
        s_I = np.sign(val_I)
        I_factorial = ifac(I)
        alpha_I = s_I * np.sqrt(I_factorial) / dsqrt * aI
        p_I = d/I_factorial * val_I**2
        return aI, val_I, alpha_I, p_I

    def compute_qmc(self):
        qmc_diag = 0
        qmc_off_diag = 0

        # loop preparation
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)

        # Diagonal terms
        for I in coeff_keys:
            # compute basic quantities
            aI = self.coeff_data[I]
            L = tuple(x + x for x in I)
            qmc_diag += aI**2 * self.haf_data[L]

        # Off-diagonal terms
        for i in range(total_I_number):
            I = coeff_keys[i]
            aI = self.coeff_data[I]
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                aJ = self.coeff_data[J]
                L = tuple(x + y for x, y in zip(I, J))
                val_L = self.haf_data[L]
                qmc_off_diag += 2 * aI * aJ * val_L

        qmc = qmc_diag + qmc_off_diag

        return qmc

    def compute_gt(self):
        gt_sum = 0
        for I, aI in self.coeff_data.items():
            gt_sum += aI * self.haf_data[I]
        return gt_sum

    def compute_qgbs_new(self, n):
        qgbs_sum = 0
        
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)
        
        for i in range(total_I_number):
            I = coeff_keys[i]
            aI, val_I, alpha_I, p_I = self.extract_short_cut(I)
            e_I = expected_sqrt_binomial(n, p_I)
            qgbs_sum += alpha_I * alpha_I * (2 * p_I - 2* np.sqrt(p_I) * e_I)
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                aJ, val_J, alpha_J, p_J = self.extract_short_cut(J)
                e = expected_value_double_sum_cached(n, p_I, p_J)
                e_J = expected_sqrt_binomial(n, p_J)
                v = e + np.sqrt(p_I) * np.sqrt(p_J) - e_I * np.sqrt(p_J) - e_J * np.sqrt(p_I) 
                qgbs_sum += 2 * alpha_I * alpha_J * v
        return qgbs_sum
        
    def compute_qgbs(self, gt, n):
        qgbs_sum_part1 = 0
        qgbs_sum_part2 = 0
        qgbs_off_diag = 0

        # loop preparation
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)

        # Diagonal terms
        for I in coeff_keys:
            # compute basic quantities
            aI, val_I, alpha_I, p_I = self.extract_short_cut(I)
            L = tuple(x + x for x in I)
            qgbs_sum_part1 += alpha_I**2 * p_I
            qgbs_sum_part2 += alpha_I * expected_sqrt_binomial(n, p_I)

        qgbs_diag = qgbs_sum_part1 - 2 * gt * qgbs_sum_part2 - gt**2

        # Off-diagonal terms
        for i in range(total_I_number):
            I = coeff_keys[i]
            aI, val_I, alpha_I, p_I = self.extract_short_cut(I)
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                aJ, val_J, alpha_J, p_J = self.extract_short_cut(I)
                qgbs_off_diag += 2 * alpha_I * alpha_J * expected_value_double_sum_optimized(n, p_I, p_J)

        qgbs = qgbs_diag + qgbs_off_diag
        return qgbs
    
    def print_log(self, path_to_log, gt, vmc_times_n, vgbs_times_n, n):
        with open(path_to_log, 'w') as f:
            # Redirect print statements to the text file
            log_message('Logging started...', f)
            log_message('========================', f)
            log_message('Basic information...', f)
            log_message(f'N: {self.N}', f)
            log_message(f'K: {self.K}', f)
            log_message(f'phi: {self.phi}', f)
            log_message('vandermonde: ', f)
            log_message(f'{self.vdet}', f)
            log_message(f'gt: {gt}', f)
            log_message(f'mc: {vmc_times_n}', f)
            log_message(f'gbs: {vgbs_times_n}', f)
            log_message(f'n: {n}', f)
            if vgbs_times_n <= vmc_times_n:
                flag_gbs_more_efficient = 1
            else:
                flag_gbs_more_efficient = 0
            log_message(f'gbs_eff: {flag_gbs_more_efficient}', f)
        return flag_gbs_more_efficient
            

