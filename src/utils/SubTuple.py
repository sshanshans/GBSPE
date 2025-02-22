import math
import numpy as npz
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from src._helpers.math import *

class SubTuple:
    """
    Class of a single tuple
    """
    def __init__(self, I, B, phi):
        self.I = I
        self.B = B
        self.phi = phi
        self.phival = None
        self.prob = None
        self.k = sum(I)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def ifac(self):
        """
        Computes the product of the factorials of each element in the tuple I
        """
        return math.prod(math.factorial(i) for i in self.I)

    def compute_phival(self):
        """
        Compute for example Haf(BI) or Haf(BI)^2
        B: gemat class
        phi: string 'haf' or 'hafsq'
        """
        if self.k == 0:
            self.phival = 1
        else:
            bmat = self.B.bmat
            if self.phi == 'haf':
                if len(self.I) != np.shape(bmat)[0]:
                    raise ValueError("The length of the tuple must be the same as the size of the Bmatrix.")
                self.phival = np.real(haf_I(bmat, self.I))
            elif self.phi == 'hafsq':
                if len(self.I) != np.shape(bmat)[0]:
                    raise ValueError("The length of the tuple must be the same as the size of the Bmatrix.")
                self.phival = np.real(haf_I(bmat, self.I))**2
            elif self.phi == 'noise':
                doubleI = self.I + self.I
                if len(doubleI) != np.shape(bmat)[0]:
                    raise ValueError("The length of the tuple must be haf of the size of the Bmatrix.")
                self.phival = np.real(haf_I(bmat, doubleI))
            else:
                raise ValueError("Invalid phi value. Must be 'haf' or 'hafsq' or 'noise'.")

    def compute_prob(self):
        """
        Computes p_I for the corresponding I
        B: gemat class
        phi: string 'haf' or 'hafsq'
        """
        if self.phival is None:
            self.compute_phival()
        if self.phi == 'haf':
            hafval = self.phival ** 2
        elif self.phi == 'hafsq':
            hafval = self.phival
        elif self.phi == 'noise':
            hafval = self.phival
        else:
            raise ValueError("Invalid phi value. Must be 'haf' or 'hafsq'.")
        pI = self.B.d * (1 / self.ifac) * hafval
        self.prob = np.real(pI)