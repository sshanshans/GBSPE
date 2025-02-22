import numpy as np
import pickle

class qCovMat:
    """
    Class of the quantum complex covariance matrix in the Gaussian state
    """
    def __init__(self, cov):
        self.cov = cov

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    def convert_covc_to_covxxpp(self, hbar=2):
        '''
        This function converts the complex covariance matrix to the real covariance matrix in xxpp ordering.
    
        Args:
            cov: A numpy array representing the complex covariance matrix.
            hbar (optional): defaulted to 2.
            
        Returns: The transformed covariance matrix in xxpp ordering.
        '''
        cov = self.cov
        n = int(np.shape(cov)[0] / 2)  # Calculate the size of the sub-matrices
        In = np.identity(n)  # Identity matrix of size n
        iIn = 1j * In  # i times the identity matrix
    
        # Constructing the block matrix W
        W = np.block([[In, iIn], [In, -iIn]])
    
        U = 1 / np.sqrt(2) * W  # Calculate U
        Udagger = np.conjugate(U.T)  # Calculate the complex conjugate transpose of U
    
        cov_xxpp = hbar * Udagger @ cov @ U  # Covariance matrix transformation
        return cov_xxpp
    
    def compute_covq(self):
        '''
        This function computes cov + 1/2 I.
    
        Args:
            cov: A numpy array representing the complex covariance matrix.
            
        Returns: covq matrix.
        '''
        cov = self.cov
        n = int(np.shape(cov)[0])  # Calculate the size of the covariance matrix
        return cov + np.identity(n)/2

## Util functions used in class
def from_xxpp_to_xpxp_transformation_matrix(d: int) -> np.ndarray:
        """
        This function is copied from Piquasso base repository.
        
        Basis changing with the basis change operator.
    
        This transformation will change the basis from xxpp-basis to xpxp-basis.
    
        .. math::
    
            T_{ij} = \delta_{j, 2i-1} + \delta_{j + 2d, 2i}
    
        Intuitively, it changes the basis as
    
        .. math::
    
            T Y = T (x_1, \dots, x_d, p_1, \dots, p_d)^T
                = (x_1, p_1, \dots, x_d, p_d)^T,
    
        which is very helpful in :mod:`piquasso._backends.gaussian.state`.
    
        Args:
            d (int): The number of modes.
    
        Returns:
            numpy.ndarray: The basis changing matrix.
        """
    
        T = np.zeros((2 * d, 2 * d), dtype=int)
        for i in range(d):
            T[2 * i, i] = 1
            T[2 * i + 1, i + d] = 1
    
        return T