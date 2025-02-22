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
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields

@dataclass
class CoeffData:
    a: float = None
    alpha: float = None
    p: float = None
    factorial: float = None
    comb: float = None

class PData:
    """
    Class to encode coefficient data and Hafnian data.
    Steps:
    1. Update B matrix and get t factor, vdet, haf_data (we also save haf_data to avoid future computation)
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
        self.dt = None
        
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

    def update_B(self, haf_folder_name, rseed=532):
        # check if haf_data already computed
        fname = f'{rseed}.pkl'
        haf_file_name = os.path.join(haf_folder_name, fname)
        if os.path.exists(haf_file_name):
            try:
                with open(haf_file_name, "rb") as f:
                    data = pickle.load(f)
                    Bmat = data['Bmat']
                    self.B = CovMat(Bmat)
                    self.vdet = data['vdet']
                    self.haf_data = data['haf_data']
                    self.t_factor = data['t']
                    self.dt = data['dt']
            except (pickle.UnpicklingError, KeyError):
                print(f'Failed to load data from {haf_file_name}')

        else:       
            Bmat = generate_random_Bmat(self.N, rseed)
            lambdas = np.linalg.eigvals(Bmat)
            self.vdet = vandermonde_determinant(lambdas)
            self.B = CovMat(Bmat)
            bmat = self.B.bmat
            start_time = time.time()
            for I in self.haf_data.keys():
                val = haf_I(bmat, I)
                self.haf_data[I] = np.real(val) # consider only real B
                # for hafsq the hafdata is hafnian without the square just like the GBSP case.
            end_time = time.time()
            print(f'total time to compute haf data table: {end_time - start_time}')
            new_B, t = self.B.compute_klevel_compatible_bmat(self.K)
            self.t_factor = t
            self.dt = new_B.d

            with open(haf_file_name, 'wb') as f:
                pickle.dump({'Bmat': Bmat, 'vdet': self.vdet, 'haf_data': self.haf_data, 't': t, 'dt': self.dt}, f)

    def update_a(self, rseed=532):
        '''
        For each I, assign the weights to on the hypersphere with uniform distribution.
        '''
        keys = list(self.coeff_data.keys())
        num_points = 1
        dimension = len(keys)
        vector = sample_points_on_sphere_real(num_points, dimension, rseed).flatten()
        for I, val in zip(keys, vector):
            self.set_value(I, 'a', val)

    # compute various quantities
    def compute_gt(self):
        start_time = time.time()
        gt_sum = 0
        for I in self.coeff_data.keys():
            aI = self.get_value(I, 'a')
            val = self.haf_data[I]
            gt_sum += aI * val
        end_time = time.time()
        print(f'total time to compute gt: {end_time - start_time}')
        return gt_sum

    def compute_qgbs(self):
        start_time = time.time()
        qgbs_sum = 0

        for I in self.coeff_data.keys():
            aI = self.get_value(I, 'a')
            qgbs_sum += aI**2 * ifac(I)

        end_time = time.time()
        print(f'total time to compute qgbs: {end_time - start_time}')
        return qgbs_sum * (self.t_factor ** (-2 * self.K)) / self.dt

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
            val = self.haf_data[L]
            qmc_diag += aI**2 * val

        # Off-diagonal terms
        for i in range(total_I_number):
            I = coeff_keys[i]
            aI = self.get_value(I, 'a')
            for j in range(i + 1, total_I_number):
                J = coeff_keys[j]
                aJ = self.get_value(J, 'a')
                L = tuple(x + y for x, y in zip(I, J))
                val_L = self.haf_data[L]
                qmc_off_diag += 2 * aI * aJ * (val_L)

        qmc = qmc_diag + qmc_off_diag
        end_time = time.time()
        print(f'total time to compute qmc: {end_time - start_time}')
        return qmc

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
            