import numpy as np
from scipy.linalg import block_diag

'''From thewalrus'''
def Qmat(cov, hbar=2):
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)
    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4
    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q

'''From thewalrus'''
def Covmat_thewalrus(Q, hbar=2):
    # number of modes
    n = len(Q) // 2
    I = np.identity(n)
    N = Q[0:n, 0:n] - I
    M = Q[n : 2 * n, 0:n]
    mm11a = 2 * (N.real + M.real) + np.identity(n)
    mm22a = 2 * (N.real - M.real) + np.identity(n)
    mm12a = 2 * (M.imag + N.imag)
    cov = np.block([[mm11a, mm12a], [mm12a.T, mm22a]])
    return (hbar / 2) * cov

'''From thewalrus'''
def Amat(cov, hbar=2, cov_is_qmat=False):
    r"""Returns the :math:`A` matrix of the Gaussian state whose hafnian gives the photon number probabilities.
    which is the bmat in our paper
    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cov_is_qmat (bool): if ``True``, it is assumed that ``cov`` is in fact the Q matrix.
    Returns:
        array: the :math:`A` matrix.
    """
    # number of modes
    N = len(cov) // 2
    X = Xmat(N)
    # inverse Q matrix
    if cov_is_qmat:
        Q = cov
    else:
        Q = Qmat(cov, hbar=hbar)
    Qinv = np.linalg.inv(Q)
    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    return A

'''From thewalrus'''
def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`

    Args:
        N (int): positive integer

    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    return X

'''From Ou et al..'''
def cov_from_T(r_array, T): # for Xanadu
    cov0 = np.diag([np.exp(2 * r) for r in r_array] + [np.exp(-2 * r) for r in r_array]);
    Q_in = Qmat(cov0)
    Tc = T.conj()
    Tt = T.T
    Th = Tt.conj()
    a = block_diag(T, Tc)
    b = block_diag(Th, Tt)
    A = (np.eye(len(T) * 2) - a @ b)
    B = a @ Q_in @ b
    Q_out = A + B
    return Covmat_thewalrus(Q_out)

def process_xanadu_samples(samples):
    samples_squeezed = np.squeeze(samples, axis=1)
    unique_samples = np.unique(samples_squeezed, axis=0)
    return [tuple(I) for I in unique_samples]