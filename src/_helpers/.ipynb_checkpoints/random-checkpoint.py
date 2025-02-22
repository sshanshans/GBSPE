import numpy as np

CONST_RANDSEED = 67215
CONST_SMALL = 1e-5
CONST_NOISELEVEL = 0.1

def generate_single_random_tuple(n, k, randseed=CONST_RANDSEED):
    """
    Generate single n tuple whose total sum is less or equal to than 2*k by
    first sample uniformly an average entry value m
    second sample n values iid using normal distribution with mean m and variance 0.2 and clip the negative entries to be 0
    third form the n values into a tuple and adjust the values to make them sum to even and
    total sum being less than 2*k
    
    Parameters:
    n (int): length of tuple
    k (int): sum of tuple should be less than 2k
    randseed (int): seed for the random number generator
    
    Returns:
    np.ndarray: Array of n random numbers
    """
    # Set the seed locally within this function
    np.random.seed(randseed)
    
    # Step 1: Calculate the average entry value at the maximum sum
    v = 2 * k // n
    
    # Step 2: Sample m uniformly between 0 and v
    m = generate_random_number_between(0, v, randseed)
    
    # Step 3: Sample n entries using normal distribution with mean m and variance 1
    s = np.random.normal(m,1,n).astype(int)
    s = np.clip(s, 0, 2 * k)
    
    # Step 4: Adjust the sum to make it even if necessary
    if np.sum(s) % 2 != 0:
        index = np.random.randint(0, n)
        s[index] += 1
    
    # Step 5: Ensure the total sum is less than 2*k
    while np.sum(s) > 2 * k:
        index = np.random.randint(0, n)
        s[index] -= 1
        if s[index] < 0:
            s[index] = 0  # Prevent negative values
    return tuple(s)

def generate_n_random_numbers_sum_to_one(n, randseed = CONST_RANDSEED):
    # Set the seed locally within this function
    np.random.seed(randseed)
    # Generate n random numbers using NumPy
    out = np.random.random(n)
    # Normalize the array so their sum is 1
    out /= out.sum()
    return out

def generate_n_random_numbers_between(n, a, b, randseed=CONST_RANDSEED):
    """
    Generate n random numbers between a and b.
    
    Parameters:
    n (int): Number of random numbers to generate
    a (float): Lower bound of the random numbers
    b (float): Upper bound of the random numbers
    randseed (int): Seed for the random number generator
    
    Returns:
    np.ndarray: Array of n random numbers between a and b
    """
    # Set the seed locally within this function
    np.random.seed(randseed)
    # Generate n random numbers between a and b using NumPy
    out = np.random.uniform(a, b, n)
    return out

def generate_random_number_between(a, b, randseed = CONST_RANDSEED):
    # Set the seed locally within this function
    np.random.seed(randseed)
    # Generate a random number between a and b using NumPy
    random_number = np.random.uniform(a, b)
    return random_number

def generate_random_Bmat(N, randseed = CONST_RANDSEED):
    """
    Generate a random valid Bmat matrix,w which is a symmetric matrix of size N x N and all eigenvalues between 0 and 1.
    """
    np.random.seed(randseed)
    random_matrix = np.random.rand(N, N)
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    adjusted_eigenvalues = generate_n_random_numbers_between(N, 0, 1, randseed)
    adjusted_matrix = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
    return adjusted_matrix
    
def generate_random_Bmat_w_positive_entries(N, randseed = CONST_RANDSEED):
    """
    Generate a random valid Bmat matrix with all entries posite
    ss note: problem is bmin can be very close to zero
    """
    random_matrix = generate_random_Bmat(N, randseed)
    # Ensure all entries are positive
    min_value = np.min(random_matrix)
    if min_value <= 0:
        random_matrix += abs(min_value) + CONST_SMALL

    # Readjust the eigenvalues by scaling the entries
    # Find the largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(random_matrix)
    max_eigenvalue = np.max(eigenvalues)
    scaled_matrix = random_matrix / (max_eigenvalue + CONST_SMALL)
    return scaled_matrix

def generate_random_Bmat_w_bounded_positive_entries(N, randseed=CONST_RANDSEED, max_attempts=10):
    """
    Generate a random valid Bmat matrix with all entries positive
    such that bmax/bmin is bounded.

    Parameters:
    N (int): Size of the square matrix.
    randseed (int): Seed for the random number generator.
    max_attempts (int): Maximum number of attempts to generate a valid matrix.
    
    Returns:
    np.ndarray: Random valid Bmat with positive entries and bmax/bmin is bounded.
    """
    # Constants
    nl = CONST_NOISELEVEL
    
    for _ in range(max_attempts):
        try:
            np.random.seed(randseed)
            # Construct a matrix with all entries 1 and then scale by 1/N
            ones_matrix = np.ones((N, N))
            scaled_ones_matrix = ones_matrix / N
            
            # Add small perturbation but keep it symmetric
            random_matrix = np.random.rand(N, N)
            symmetric_matrix = (random_matrix + random_matrix.T) / 2
            scaled_matrix = nl/N * symmetric_matrix
            random_matrix = scaled_ones_matrix + scaled_matrix
            eigenvalues, eigenvectors = np.linalg.eigh(random_matrix)
            adjusted_eigenvalues = np.clip(eigenvalues, CONST_SMALL, 1-CONST_SMALL)
            adjusted_matrix = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
            
            # Check if all eigenvalues are strictly greater than 0
            if np.min(adjusted_matrix > 0):
                return adjusted_matrix
            else:
                print('Attempt', _+1, ': Negative eigenvalues encountered. Readjusting the random matrix...')
                # Use a different random seed for the next attempt
                randseed += 1
                nl /= 2 
        except Exception as e:
            print('Error occurred:', str(e))
    
    print(f"Failed to generate a valid matrix after {max_attempts} attempts.")
    return None

def sample_points_on_sphere_real(num_points, dimension, randseed = CONST_RANDSEED):
    """
    Generate uniformly distributed points on the surface of an (n-1)-dimensional sphere.
    
    Parameters:
    - num_points: int, number of points to generate
    - dimension: int, dimension of the sphere (e.g., for a 3D sphere, dimension=3)
    
    Returns:
    - points: np.ndarray, shape (num_points, dimension), each row is a point on the sphere
    """
    # Set the seed locally within this function
    np.random.seed(randseed)
    
    # Sample points from a standard normal distribution
    points = np.random.normal(0, 1, (num_points, dimension))
    
    # Normalize each point to have unit norm
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points

def sample_points_uniform(num_points, dimension, randseed = CONST_RANDSEED):
    """
    Generate uniformly distributed points on the surface of an (n-1)-dimensional sphere.
    
    Parameters:
    - num_points: int, number of points to generate
    - dimension: int, dimension of the sphere (e.g., for a 3D sphere, dimension=3)
    
    Returns:
    - points: np.ndarray, shape (num_points, dimension), each row is a point on the sphere
    """
    # Set the seed locally within this function
    np.random.seed(randseed)
    
    # Sample points from a standard normal distribution
    points = np.random.uniform(-1, 1, (num_points, dimension))
    
    return points

def generate_rand_coeff():
    np.random.seed(None)
    return np.random.uniform(-1, 1)