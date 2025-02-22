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
from dataclasses import dataclass, fields

@dataclass
class CoeffData:
    a: float = None
    alpha: float = None
    p: float = None
    factorial: float = None
    comb: float = None

class EData:
    """
    Class to encode coefficient data and Hafnian data.
    Steps:
    1. Update B matrix and get t factor, vdet, haf_data
    2. Update coeff data, first obtain factorial, p and comb
    3. Sample on unit sphere and multiply with t factor to get a, further obtain alpha,
    4. Compute gt, qmc, and qgbs.
    """
    def __init__(self, N, K, phi, cdict, hdict):
        # Validation
        if not (isinstance(N, int) and N > 0):
            raise ValueError("N must be a positive integer")
        if not (isinstance(K, int) and K > 0):
            raise ValueError("K must be a positive integer")
        if phi not in ['haf']:
            raise ValueError("phi must be set to 'haf'.")

        self.N = N
        self.K = K
        self.phi = phi
        
        self.B = None
        self.vdet = None 
        self.haf_data = hdict
        self.t_factor = None
        
        self.comb_data_double = None
        self.coeff_data = {
            key: CoeffData()
            for key in cdict.keys()
        }

    def __reduce__(self):
        # Return a tuple with the class, the arguments to pass to __init__, and the instance's state
        return (self.__class__, (self.N, self.K, self.phi, 
                                 {key: vars(data) for key, data in self.coeff_data.items()},
                                 self.haf_data), self.__dict__)

    def __setstate__(self, state):
        # Update the instance's __dict__ with the unpickled state
        self.__dict__.update(state)

    def save(self, filename):
        """Save the current object to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load an object from a file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def set_value(self, key, field_name, value):
        """Set a specific field value for a key."""
        if key in self.coeff_data:
            # Validate field_name exists in CoeffData
            if field_name in {f.name for f in fields(CoeffData)}:
                setattr(self.coeff_data[key], field_name, value)
            else:
                raise ValueError(f"Invalid field name '{field_name}'.")
        else:
            raise KeyError(f"Key '{key}' not found in coeff_data.")

    def get_value(self, key, field_name):
        """Get a specific field value for a key."""
        if key in self.coeff_data:
            # Validate field_name exists in CoeffData
            if field_name in {f.name for f in fields(CoeffData)}:
                return getattr(self.coeff_data[key], field_name)
        return None

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
        self.t_factor = t**(-self.K)

    def update_B_followup(self, n=200, pflag=True):
        start_time = time.time()
        self.update_factorial_p_comb(n)
        self.update_comb_double_data(n, pflag)
        end_time = time.time()
        print(f'total time to compute combinatorial factors: {end_time - start_time}')

    def update_B_followup2(self):
        start_time = time.time()
        d = self.B.d
        for I in self.coeff_data.keys():
            I_factorial = ifac(I)
            self.set_value(I, 'factorial', I_factorial)
            val_I = self.haf_data[I]
            p_I = d/I_factorial * val_I**2
            self.set_value(I, 'p', p_I)
        end_time = time.time()
        print(f'total time to compute all p_I: {end_time - start_time}')

    def update_B_followup3(self):
        start_time = time.time()
        d = self.B.d
        for I in self.coeff_data.keys():
            I_factorial = ifac(I)
            self.set_value(I, 'factorial', I_factorial)
        end_time = time.time()
        print(f'total time to compute all p_I: {end_time - start_time}')
        
    def update_factorial_p_comb(self, n=200):
        d = self.B.d
        for I in self.coeff_data.keys():
            I_factorial = ifac(I)
            self.set_value(I, 'factorial', I_factorial)
            val_I = self.haf_data[I]
            p_I = d/I_factorial * val_I**2
            self.set_value(I, 'p', p_I)
            e_I = expected_sqrt_binomial(n, p_I)
            self.set_value(I, 'comb', e_I)
        
    def update_comb_double_data(self, n=200, pflag=True):
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)
        self.comb_data_double = {key: {} for key in coeff_keys}
        
        if pflag:
            # Precompute p_I values in the main process
            p_values = {key: self.get_value(key, 'p') for key in coeff_keys}
            # Create tasks with precomputed p_I and p_J
            tasks = [
                (n, p_values[coeff_keys[i]], p_values[coeff_keys[j]])
                for i in range(total_I_number)
                for j in range(i + 1, total_I_number)
            ]
        
            max_workers = 30
        
            # Use ProcessPoolExecutor to execute tasks in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pair = {
                    executor.submit(expected_value_double_sum_cached, n, p_I, p_J): (coeff_keys[i], coeff_keys[j])
                    for (n, p_I, p_J), (i, j) in zip(tasks, [(i, j) for i in range(total_I_number) for j in range(i + 1, total_I_number)])
                }
        
                for future in as_completed(future_to_pair):
                    I, J = future_to_pair[future]
                    e = future.result()
                    self.comb_data_double[I][J] = e
        
        else:
            for i in range(total_I_number):
                I = coeff_keys[i]
                p_I = self.get_value(I, 'p')
                for j in range(i + 1, total_I_number):
                    J = coeff_keys[j]
                    p_J = self.get_value(J, 'p')
                    #e = expected_value_cpp.expected_value_double_sum_cpp(n, p_I, p_J)
                    e = expected_value_double_sum_cached(n, p_I, p_J)
                    self.comb_data_double[I][J] = e


    def update_a_alpha(self, rseed=532):
        '''
        For each I, assign the weights to on the hypersphere with uniform distribution.
        '''
        d = self.B.d
        dsqrt = np.sqrt(d)
        keys = list(self.coeff_data.keys())
        num_points = 1
        dimension = len(keys)
        #vector = sample_points_uniform(num_points, dimension, rseed).flatten()
        vector = sample_points_on_sphere_real(num_points, dimension, rseed).flatten()
        for I, val in zip(keys, vector):
            aI = val * self.t_factor
            self.set_value(I, 'a', aI)
            val_I = self.haf_data[I]
            s_I = np.sign(val_I)
            I_factorial = self.get_value(I, 'factorial')
            alpha_I = s_I * np.sqrt(I_factorial) / dsqrt * aI
            self.set_value(I, 'alpha', alpha_I)

    def update_a_alpha2(self, rseed=532):
        '''
        For each I, assign the weights to on the hypersphere with uniform distribution.
        '''
        d = self.B.d
        dsqrt = np.sqrt(d)
        keys = list(self.coeff_data.keys())
        num_points = 1
        dimension = len(keys)
        #vector = sample_points_uniform(num_points, dimension, rseed).flatten()
        vector = sample_points_on_sphere_real(num_points, dimension, rseed).flatten()
        for I, val in zip(keys, vector):
            aI = val * self.t_factor
            self.set_value(I, 'a', aI)
            I_factorial = self.get_value(I, 'factorial')
            alpha_I = np.sqrt(I_factorial) / dsqrt * aI
            self.set_value(I, 'alpha', alpha_I)

    def compute_qmc(self):
        start_time = time.time()
        qmc_diag = 0
        qmc_off_diag = 0

        # loop preparation
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)

        # Diagonal terms
        for I in coeff_keys:
            # compute basic quantities
            aI = self.get_value(I, 'a')
            L = tuple(x + x for x in I)
            qmc_diag += aI**2 * self.haf_data[L]

        # Off-diagonal terms
        for i in range(total_I_number):
            I = coeff_keys[i]
            aI = self.get_value(I, 'a')
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                aJ = self.get_value(J, 'a')
                L = tuple(x + y for x, y in zip(I, J))
                val_L = self.haf_data[L]
                qmc_off_diag += 2 * aI * aJ * val_L

        qmc = qmc_diag + qmc_off_diag
        end_time = time.time()
        print(f'total time to compute qmc: {end_time - start_time}')
        return qmc

    def compute_gt(self):
        start_time = time.time()
        gt_sum = 0
        for I in self.coeff_data.keys():
            aI = self.get_value(I, 'a')
            gt_sum += aI * self.haf_data[I]

        end_time = time.time()
        print(f'total time to compute gt: {end_time - start_time}')
        return gt_sum

    def compute_qgbs_new(self):
        start_time = time.time()
        qgbs_sum = 0
        
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)
        
        for i in range(total_I_number):
            I = coeff_keys[i]
            alpha_I = self.get_value(I, 'alpha')
            p_I = self.get_value(I, 'p')
            e_I = self.get_value(I, 'comb')
            qgbs_sum += alpha_I * alpha_I * (2 * p_I - 2* np.sqrt(p_I) * e_I)
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                alpha_J = self.get_value(J, 'alpha')
                p_J = self.get_value(J, 'p')
                e_J = self.get_value(J, 'comb')
                e = self.comb_data_double[I][J]
                v = e + np.sqrt(p_I) * np.sqrt(p_J) - e_I * np.sqrt(p_J) - e_J * np.sqrt(p_I) 
                qgbs_sum += 2 * alpha_I * alpha_J * v

        end_time = time.time()
        print(f'total time to compute qgbs: {end_time - start_time}')
        return qgbs_sum

    def compute_qgbs_new2(self):
        start_time = time.time()
        qgbs_sum = 0
        
        coeff_keys = list(self.coeff_data.keys())
        total_I_number = len(coeff_keys)
        
        for i in range(total_I_number):
            I = coeff_keys[i]
            alpha_I = self.get_value(I, 'alpha')
            p_I = self.get_value(I, 'p')
            eII = 1/4 * (1 - p_I)
            qgbs_sum += alpha_I * alpha_I * eII
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                alpha_J = self.get_value(J, 'alpha')
                p_J = self.get_value(J, 'p')
                eIJ = -1/4 * np.sqrt(p_I) * np.sqrt(p_J)
                qgbs_sum += 2 * alpha_I * alpha_J * eIJ

        end_time = time.time()
        print(f'total time to compute qgbs: {end_time - start_time}')
        return qgbs_sum

    def compute_qgbs_new3(self):
        start_time = time.time()
        qgbs_sum = 0

        for I in self.coeff_data.keys():
            alpha_I = self.get_value(I, 'alpha')
            qgbs_sum += alpha_I**2

        end_time = time.time()
        print(f'total time to compute qgbs: {end_time - start_time}')
        return qgbs_sum
    
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

    def print_log2(self, path_to_log, gt, vmc, vgbs):
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
            log_message(f'mc: {vmc}', f)
            log_message(f'gbs: {vgbs}', f)
            if vgbs <= vmc:
                flag_gbs_more_efficient = 1
            else:
                flag_gbs_more_efficient = 0
            log_message(f'gbs_eff: {flag_gbs_more_efficient}', f)
        return flag_gbs_more_efficient
            

