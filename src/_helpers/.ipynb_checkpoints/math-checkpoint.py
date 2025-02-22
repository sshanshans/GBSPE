import numpy as np
import itertools
import math
from thewalrus._hafnian import hafnian, hafnian_repeated
from pathlib import Path
from scipy.special import comb
from scipy.stats import binom

CONST_PI = math.pi
CONST_DOUBLEPI = 2 * math.pi
CONST_EXP = np.exp(1/25 - 1/6)
CONST_SQRT_PI = 1/np.sqrt(math.pi)

def haf(B):
    return hafnian(B)
    
def hafsq(B):
    return hafnian(B)**2

def haf_I(B, I):
    return hafnian_repeated(B, I)

def q_minus(q):
    return min(q, 0)

def compute_haf(lambdas, I):
    lambdas = np.array(lambdas)
    I = np.array(I)
    if np.any(I % 2 != 0):
        return 0
    powers = lambdas ** (I / 2)
    double_factorials = np.array([np.prod(range(i - 1, 0, -2)) if i > 0 else 1 for i in I])
    result = np.prod(powers * double_factorials)
    return result

def product_of_powers_double(values, indices):
    """
    Computes the product of each value in 'values' raised to the corresponding power in 'indices'.

    Args:
    values (list): A list of values [x1, x2, ..., xn, xn+1, ..., x2n].
    indices (list): A list of indices [i1, i2, ..., in].

    Returns:
    float: The result of x1^(i1) * x2^(i2) * ... * xn^(in) * xn+1^(i1) * ... * x2n^(in).
    """
    doubled_indices = indices + indices
    
    if len(values) != len(doubled_indices):
        raise ValueError("The length of values and doubled indices lists must be the same.")

    result = 1
    for value, index in zip(values, doubled_indices):
        result *= value ** index

    return result

def product_of_powers_single(values, indices):
    """
    Computes the product of each value in 'values' raised to the corresponding power in 'indices'.

    Args:
    values (list): A list of values [x1, x2, ..., xn].
    indices (list): A list of indices [i1, i2, ..., in].

    Returns:
    float: The result of x1^(i1) * x2^(i2) * ... * xn^(in).
    """
    if len(values) != len(indices):
        raise ValueError("The length of values and indices lists must be the same.")

    result = 1
    for value, index in zip(values, indices):
        result *= value ** index

    return result

def polylog(x, s, K):
    """
    Computes the polylogarithm function up to finite sum K and order s
    Args:
    x (float): funciton input
    s (float): order of the polylog
    K (int): truncation
    
    Returns: 
    float: function value
    """
    if K < 0:
        raise ValueError("K must be a non-negative integer")
    
    if K == 0:
        return 0
    
    k = np.arange(1, K + 1)
    terms = (x**k) / (k**s)
    return np.sum(terms)

def Hi(x, q, N):
    """
    Computes hyperbolic function of finite sum up to 2k = N or (N-1 if N is odd) without the constant 1. This corresbonds to Hi_{q, N/2} in the paper.
    Args:
    x (float): funciton input
    q (float): similar to polylog s
    N (int): truncation level as given in the paper
    
    Returns: 
    float: function value
    """
    k = np.arange(1, N//2 + 1)
    # Vectorize the math.factorial function to apply it to each element
    factorial_vectorized = np.vectorize(math.factorial)
    terms = x**(2*k) * pow(k, q) / factorial_vectorized(2*k)
    return np.sum(terms)

def Gqk(z, q, N, K):
    """
    The G_q, K(z, N) function defined as in the paper
    
    Args:
    z (float)
    q (float)
    N (int)
    K (int)
    
    Returns: function value (float)
    """
    sk = compute_Sk(N, K)
    return Hi(z, q, N) +  2**((N-1)/2 - q_minus(q)) * CONST_PI**((N-1)/2) * N**(q - 1/2) * np.exp(N/13) * polylog(2*z/N, 0, N) * polylog( (z/N)**N, 1/2-N/2-q, sk)

def Rqk(z, q, K):
    """
    The R_q,K(z, N) function defined as in the paper
    
    Args:
    z (float)
    q (float)
    K (int)
    
    Returns: function value (float)
    """
    c = CONST_SQRT_PI * 1/2
    if q >= 0:
        return c * 2**(-q) * polylog(z, 1/2 - q, K)
    else:
        return c * polylog(z, 1/2 - 2*q, K)
    
def compute_Sk(N, K):
    """
    Computes SK such that 2K = N * SK + RK where RK is between 1 and N
    
    Args:
    N (int)
    K (int)
    
    Returns: 
    float: SK
    """
    RK = 2*K % N
    SK = 2*K // N
    if RK == 0:
        if SK >=1:
            SK = SK-1
            RK = N
    return SK

def compute_mk(N, k):
    """
    Computes mk defined as in the paper
    mk = (sk!)^N * (sk + 1)^rk
    with 2k = N * sk + rk
    
    Args:
    N (int)
    k (int)
    
    Returns: 
    float: mk
    """
    RK = 2*k % N
    SK = 2*k // N
    if RK == 0:
        if SK >=1:
            SK = SK-1
            RK = N
    return (math.factorial(SK))**N * (SK + 1)**RK

def compute_c1(z, q, K):
    """
    Computes c1 defined as in the paper
    
    Args:
    z (float)
    q (float) as defined in the epxression; this is usually 0
    K (int)
    
    Returns: 
    c1 (float)
    """
    return 1 + CONST_SQRT_PI * CONST_EXP * polylog(z, 1/2 -q, K)

def compute_c2(z, q, K):
    """
    Computes c2 defined as in the paper
    
    Args:
    z (float)
    q (float) as defined in the expression
    K (int)
    
    Returns: 
    c2 (float)
    """
    return 1 + CONST_SQRT_PI * polylog(z, 1/2-q, K)

def ifac(I):
    """
    Computes the product of the factorials of each element in the tuple I
    """
    return math.prod(math.factorial(i) for i in I)

def vandermonde_determinant(lambdas, real_bool=True):
        n = len(lambdas)
        det = 1
        for i in range(n):
            for j in range(i + 1, n):
                det *= (lambdas[j] - lambdas[i])
        if real_bool:
            det = np.abs(det)
        else: 
            raise ValueError("real_bool must be set to be True")
        return det

def expected_sqrt_binomial(n, p):
    """
    Computes the expected value of sqrt(x/n) where x is a binomial(n, p) random variable.

    Args:
        n (int): Number of trials.
        p (float): Probability of success in each trial.

    Returns:
        float: The expected value of sqrt(x/n).
    """
    expected_value = sum(math.sqrt(x / n) * binom.pmf(x, n, p) for x in range(1, n + 1))
    return expected_value

def expected_sqrt_double_sum(n, p1, p2):
    result = 0.0
     # Precompute terms for efficiency
    q = p1 / (1 - p2) 
    base = (1 - q) / (1 - p1) 
    for l in range(1, n + 1):
        val1 = math.sqrt(l / n) * binom.pmf(l, n, p1)
        val2 = np.power(base, n - l)
        for j in range(1, n - l + 1):
            val3 =  math.sqrt(j / n) * binom.pmf(j, n-l, p2)
            val4 = np.power(1-q, -j)
            result += val1 * val2 * val3 * val4
    return result

def expected_sqrt_double_sum_vec(n, p1, p2):
    # Precompute constants
    q = p1 / (1 - p2)
    base = (1 - q) / (1 - p1)

    # Precompute binomial PMFs for efficiency
    binom_pmf_p1 = binom.pmf(np.arange(1, n + 1), n, p1)  # Binom PMF for l
    sqrt_l = np.sqrt(np.arange(1, n + 1) / n)  # sqrt(l/n) for l
    result = 0.0

    # Outer loop for l
    for l in range(1, n + 1):
        val1 = sqrt_l[l - 1] * binom_pmf_p1[l - 1]
        val2 = np.power(base, n - l)
        
        # Inner loop for j
        j_vals = np.arange(1, n - l + 1)  # Range of j
        sqrt_j = np.sqrt(j_vals / n)  # sqrt(j/n)
        binom_pmf_p2 = binom.pmf(j_vals, n - l, p2)  # Binom PMF for j
        val3 = sqrt_j * binom_pmf_p2
        val4 = np.power(1 - q, -j_vals)

        # Accumulate result using vectorized computation
        result += val1 * val2 * np.sum(val3 * val4)

    return result
    
def expected_value_double_sum(n, p1, p2):
    """
    Computes the expected value for the given double summation.

    Args:
        n (int): Total number of trials.
        p1 (float): Probability for the first event.
        p2 (float): Probability for the second event.

    Returns:
        float: The computed expected value.
    """
    result = 0.0
    for l in range(1, n + 1):
        for j in range(1, n - l + 1):
            coeff = comb(n, l) * comb(n - l, j)  # Compute n!/(l!j!(n-l-j)!)
            prob = coeff * (p1**l) * (p2**j) * ((1 - p1 - p2)**(n - l - j))
            result += math.sqrt(l * j / n**2) * prob
    return result

def expected_value_double_sum_optimized(n, p1, p2):
    """
    Computes the double summation more efficiently.

    Args:
        n (int): Total number of trials.
        p1 (float): Probability of the first event.
        p2 (float): Probability of the second event.

    Returns:
        float: The computed expected value.
    """
    result = 0.0

    # Precompute binomial coefficients for all l
    binom_n_l = np.array([comb(n, l) for l in range(1, n + 1)])

    for l in range(1, n + 1):
        max_j = n - l
        j_vals = np.arange(1, max_j + 1)

        # Precompute binomial coefficients for (n-l, j)
        binom_n_l_j = np.array([comb(n - l, j) for j in j_vals])

        # Compute probabilities and summand terms
        coeffs = binom_n_l[l - 1] * binom_n_l_j
        probs = coeffs * (p1**l) * (p2**j_vals) * ((1 - p1 - p2)**(n - l - j_vals))
        summands = np.sqrt(l * j_vals / n**2) * probs

        # Sum over j values for this l
        result += np.sum(summands)

    return result

# Global cache for binomial coefficients
_binom_cache = {}

def get_binom_cache(n):
    """
    Fetch or compute the binomial coefficients for a given n and cache them.

    Args:
        n (int): Total number of trials.

    Returns:
        np.ndarray: Cached binomial coefficients for n.
    """
    if n not in _binom_cache:
        # Compute and store binomial coefficients for all l in 1 to n
        _binom_cache[n] = np.array([comb(n, l) for l in range(1, n + 1)])
    return _binom_cache[n]

def expected_value_double_sum_cached(n, p1, p2):
    """
    Computes the double summation more efficiently using cached binomial coefficients.

    Args:
        n (int): Total number of trials.
        p1 (float): Probability of the first event.
        p2 (float): Probability of the second event.

    Returns:
        float: The computed expected value.
    """
    # Fetch binomial coefficients for l
    binom_n_l = get_binom_cache(n)
    
    # Initialize result
    result = 0.0

    # Iterate over l
    for l in range(1, n + 1):
        max_j = n - l
        j_vals = np.arange(1, max_j + 1)

        # Precompute binomial coefficients for (n-l, j)
        binom_n_l_j = comb(n - l, j_vals)

        # Compute all terms in one vectorized step
        coeffs = binom_n_l[l - 1] * binom_n_l_j
        probs = coeffs * (p1**l) * (p2**j_vals) * ((1 - p1 - p2)**(n - l - j_vals))
        summands = np.sqrt(l * j_vals / n**2) * probs

        # Sum over j values for this l
        result += summands.sum()

    return result