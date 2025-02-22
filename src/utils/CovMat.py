import numpy as np
import pickle
from src.utils.qCovMat import qCovMat
from src._helpers import optimize

class CovMat:
    """
    Class of the B matrix, which corresponds to 
    the covariance matrix in the Gaussian Expectation problem
    Sometimes the B matrix is only the upperleft diagonal block

    In this special class, we do not check the eigenvalues as there is no direct Gaussian expectation values problem
    """
    def __init__(self, bmat):
        self.bmat = bmat
        self.dinv = self.compute_dinv()
        self.d = self.compute_d()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def bmin(self):
        return np.min(self.bmat)

    @property
    def bmax(self):
        return np.max(self.bmat)

    def check_eigenvalues(self):
        eigenvalues = np.linalg.eigvals(self.bmat)
        if not np.all((0 < eigenvalues) & (eigenvalues < 1)):
            raise ValueError("The eigenvalues of the matrix are not between 0 and 1.")
        return eigenvalues

    def check_symmetry(self):
        if not np.allclose(self.bmat, self.bmat.T):
            raise ValueError("The matrix is not symmetric.")

    def compute_d(self):
        return 1/self.dinv
        
    # def compute_dinv(self):
    #     """
    #     Compute d inverse by the following
    #     Converts a given B matrix to a complex covariance matrix of a Gaussian state
    #     Returns: 1/d
    #     """
    #     bmat = self.bmat
    #     n = np.shape(bmat)[0] # which is twice of the quantum modes
    #     I = np.eye(int(n/2))
    #     Z = np.zeros_like(I)
    #     permute_block_matrix = np.block([[Z, I], [I, Z]])

    #     S = permute_block_matrix @ bmat 
    #     inv_sigma_q = np.eye(n) - S
    
    #     # Calculate the sigma_q matrix by taking the inverse of inv_sigma_q
    #     #sigma_q = np.linalg.inv(inv_sigma_q)
    
    #     return 1/np.sqrt(np.linalg.det(inv_sigma_q))

    def compute_dinv(self):
        """
        compute dinv
        """
        covc = qCovMat(self.convert_bmat_to_covc())
        covq = covc.compute_covq()
        det_covq = np.linalg.det(covq)
        if det_covq <= 0:
            raise ValueError("The determinant of covq must be positive.")
        d_inv = np.sqrt(det_covq)
        return np.real(d_inv)
        
    def convert_bmat_to_covc(self):
        """
        Converts a given B matrix to a complex covariance matrix of a Gaussian state
        alpha and alpha dagger representation

        Only suitable when phi = hafsq or haf
    
        Returns:
        np.ndarray: The computed complex covariance matrix.
        """
        bmat = self.bmat
        n = np.shape(bmat)[0]
    
        # Calculate the complex conjugate of bmat
        bmat_conj = np.conjugate(bmat)
    
        # Construct the inverse of the sigma_q matrix
        identity_block = np.eye(n)
        top_right_block = -bmat_conj
        bottom_left_block = -bmat
        upper_block = np.concatenate((identity_block, top_right_block), axis=1)
        lower_block = np.concatenate((bottom_left_block, identity_block), axis=1)
        inv_sigma_q = np.concatenate((upper_block, lower_block), axis=0)
    
        # Calculate the sigma_q matrix by taking the inverse of inv_sigma_q
        sigma_q = np.linalg.inv(inv_sigma_q)
    
        # Subtract 1/2 from the diagonal terms to get the final sigma matrix
        sigma = sigma_q - 0.5 * np.eye(2 * n)
    
        # Return the final sigma matrix
        return sigma
        
    def convert_bmat_to_cov_normal(self):
        """
        Convert the n by n matrix Bmat to the 2n by 2n covariance matrix of the multivariate normal distribution
        cov = Bmat oplus Bmat 

        Only suitable when phi = hafsq. 
    
        Returns:
        (np.ndarray): cov
        """
        bmat = self.bmat
        # Extract the matrix size n
        n = int(np.shape(bmat)[0]) 
        # Construct the building blocks
        zero_block = np.zeros((n,n))
        # Constructing the block covariance matrix
        cov = np.block([[bmat, zero_block], [zero_block, bmat]])
        return cov

    def compute_master_bmat_det(self, scale=None):
        """
        Compute the sum of 1/I! Haf(BI)^2 as given in the master theorem

        Only suitable when phi = hafsq. 
        """
        B = self.bmat
        n = B.shape[0]  # Size of the square matrix B
        I = np.eye(2 * n)  # Identity matrix of size 2n x 2n
        
        # Construct the block matrix [0, B; B, 0]
        Z = np.zeros_like(B)
        block_matrix = np.block([[Z, B], [B, Z]])

        if scale is None:
            # Calculate A = I - [0, B; B, 0]
            A = I - block_matrix
        else:
            A = I - (np.eye(2*n)*scale)@block_matrix
        
        # Compute the determinant of A
        determinant = np.linalg.det(A)
        return 1/np.sqrt(determinant)

    def compute_mean_photon(self):
        """
        Compute the mean phonton number for each mode, to output the mean photon number from the entire tuple, simply take the sum
        """
        sigma = self.convert_bmat_to_covc()
        de = extract_submatrix_diagonal(sigma) - 0.5
        return de

    def compute_klevel_compatible_bmat(B, k, print_flag = False):
        """Process the CoeffDict T to make it compatible for the klevel method"""
        bmat = B.bmat
        lambdas = B.check_eigenvalues()
        optimized_t, final_loss = optimize.simple_grid_search(lambdas, k, n=int(1e4))
        print(f"Optimized t: {optimized_t:.6f}, Final loss: {final_loss:.6f}")
        bmat_scaled = bmat * optimized_t
        B_scaled = CovMat(bmat_scaled)
        original_mean_photon = B.compute_mean_photon()
        print('original mean photon number', np.sum(original_mean_photon))
        new_mean_photon = B_scaled.compute_mean_photon()
        print('new mean photon number', np.sum(new_mean_photon))
        if print_flag:
            print('==========================================')
            print('original bmat', bmat)
            print('original sum', np.sum(original_mean_photon))
            print('new bmat', bmat_scaled)
            new_mean_photon = B_scaled.compute_mean_photon()
            print('new mean photon number', new_mean_photon)
            print('new sum', np.sum(new_mean_photon))
            print('==========================================')
        return B_scaled, optimized_t

def extract_submatrix_diagonal(matrix):
    """Extract the top-left n x n submatrix and return its diagonal elements."""
    # Determine n based on the input matrix shape (assumes it's square and of size 2n x 2n)
    n = matrix.shape[0] // 2
    
    # Extract the top-left n x n submatrix
    submatrix = matrix[:n, :n]
    
    # Extract the diagonal elements of the submatrix
    diagonal_elements = np.diag(submatrix)
    
    return diagonal_elements
