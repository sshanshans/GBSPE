import pickle
import os
import numpy as np
import uuid
from os.path import join
from multiprocessing import Pool
from src._helpers.check import check_and_create_folder

class Estimator():
    def __init__(self, T, path_to_samples):
        self.T = T
        self.path_to_samples = path_to_samples
        self.samples = None
        self.estimates = []
        self.processing_method = None

    def set_processing_method(self, method):
        self.processing_method = method

    def run_sampling(self, num_samples, num_tasks):
        '''
        num_samples (int): number of samples to simulate
        num_tasks (int): distribute for parallel processing
        '''
        batch_size = int(np.ceil(num_samples / num_tasks))
        print(f"Number of tasks to be processed: {num_tasks}")
        print(f"Number of samples in each task: {batch_size}")
        T = self.T
        processing_method = self.processing_method
        if processing_method is None:
            raise ValueError("Processing method is not set")
        if self.processing_method.name not in ["GBSI", "MC", "GBSS", "MCS"]:
            raise ValueError("Processing method must be GBSI, GBSS or MC, MCS")
        if self.processing_method.name in ["GBSS", "MCS"] and num_tasks > 1:
            raise ValueError("No parallel processing can be used for GBS and MCS")

        if self.processing_method.name in ["GBSS", "MCS"]:
            result = processing_method.process(T, batch_size)
            self.save_samples(result)
        else:
            with Pool() as pool:
                tasks = [(T, batch_size, processing_method) for _ in range(num_tasks)]
                for i, result in enumerate(pool.imap_unordered(_process_single_task, tasks)):
                    self.save_samples(result)
                    print(f"Step {i+1}/{num_tasks}")

    def run_sampling_noise(self, path_to_samples, param=None):
        '''
        path_to_samples (string): samples saved by other methods
        '''
        T = self.T
        processing_method = self.processing_method
        if processing_method is None:
            raise ValueError("Processing method is not set")
        if self.processing_method.name not in ["GBSIn", "GBSIs"]:
            raise ValueError("Processing method must be GBSIn or GBSIs")

        if os.path.exists(path_to_samples):
            for filename in os.listdir(path_to_samples):
                if filename.endswith('.npy'):
                    filepath = os.path.join(path_to_samples, filename)
                    xs = np.load(filepath)
                    if self.processing_method.name == "GBSIn":
                        es = processing_method.process(T, xs)
                    else:
                        es = processing_method.process(T, xs, r=param)
                    self.save_samples(es)
        else:
            print(f"Path {path_to_samples} does not exist.")

    def save_samples(self, result):
        try:
            check_and_create_folder(self.path_to_samples)
            path_to_file = join(self.path_to_samples, f'{uuid.uuid4()}.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Error occurred while saving samples: {e}")

    def load_samples(self, path_to_samples):
        self.samples = []  # Clear current samples
        if os.path.exists(path_to_samples):
            for filename in os.listdir(path_to_samples):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(path_to_samples, filename)
                    with open(filepath, 'rb') as f:
                        self.samples.extend(pickle.load(f))
        else:
            print(f"Path {path_to_samples} does not exist.")

    def is_samples_directory_empty(self):
        if not os.path.exists(self.path_to_samples):
            return True
        return not any(fname.endswith('.pkl') for fname in os.listdir(self.path_to_samples))
        
    def compute_estimates(self, num_threads, thread_size):
        if self.processing_method is None:
            raise ValueError("Processing method is not set")
        
        self.estimates = [] # reset estimates
        
        if self.is_samples_directory_empty():
            self.run_sampling(thread_size * num_threads, num_threads)
        self.load_samples(self.path_to_samples)
        if len(self.samples) < thread_size * num_threads:
            raise ValueError("Not enough samples to compute estimates")
        
        for i in range(num_threads):
            sample_set = self.samples[i * thread_size:(i + 1) * thread_size]
            estimate = []
            cumulative_sum = 0
            for n, s in enumerate(sample_set):
                cumulative_sum += s
                estimate.append(cumulative_sum / (n + 1))
            self.estimates.append(estimate)

    def compute_estimates_thinning(self, num_threads, thread_size, step_size=100):
        if self.processing_method is None:
            raise ValueError("Processing method is not set")
        
        self.estimates = [] # reset estimates
        
        if self.is_samples_directory_empty():
            self.run_sampling(thread_size * num_threads, num_threads)
        self.load_samples(self.path_to_samples)
        if len(self.samples) < thread_size * num_threads:
            raise ValueError("Not enough samples to compute estimates")
        
        for i in range(num_threads):
            sample_set = self.samples[i * thread_size:(i + 1) * thread_size]
            estimate = []
            cumulative_sum = 0
            for n, s in enumerate(sample_set):
                cumulative_sum += s
                if (n + 1) % step_size == 0:
                    estimate.append(cumulative_sum / (n + 1))
            if len(sample_set) % step_size != 0:
                estimate.append(cumulative_sum / len(sample_set))
            self.estimates.append(estimate)
                
    def compute_multiplicative_errors(self,  gt=None):
        if self.estimates is None:
            raise ValueError("Estimates have not been computed yet")
        if gt is None:
            gt = self.T.gt
        return np.abs(np.array(self.estimates) - gt)/ np.abs(gt)

    def compute_additive_errors(self, gt=None):
        if self.estimates is None:
            raise ValueError("Estimates have not been computed yet")
        if gt is None:
            gt = self.T.gt
        return np.abs(np.array(self.estimates) - gt)

    def save_estimates(self, path_to_estimates, gt=None):
        try:
            check_and_create_folder(path_to_estimates)
            # save estimate vals
            path_to_file = join(path_to_estimates, 'val_est.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.estimates, f)
            # save multiplicative error
            path_to_file = join(path_to_estimates, 'mul_err.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.compute_multiplicative_errors(gt), f)
            # save multiplicative error
            path_to_file = join(path_to_estimates, 'add_err.pkl')
            with open(path_to_file, 'wb') as f:
                pickle.dump(self.compute_additive_errors(gt), f)
        except Exception as e:
            print(f"Error occurred while saving samples: {e}")
        
def _process_single_task(args):  
    T, batch_size, processing_method = args
    if processing_method is None:
        raise ValueError("Processing method is not set")
    return processing_method.process(T, batch_size)