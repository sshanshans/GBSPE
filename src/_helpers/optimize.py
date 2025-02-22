import numpy as np

def objective_function(t, lambdas, k):
    f_t = t**2 * np.sum(lambdas**2 / (1 - t**2 * lambdas**2))
    return (f_t - 2 * k)**2

def gradient(t, lambdas, k):
    f_t = t**2 * np.sum(lambdas**2 / (1 - t**2 * lambdas**2))
    df_t = 2 * t * np.sum(lambdas**2 / (1 - t**2 * lambdas**2)) + t**3 * np.sum(2 * lambdas**4 / (1 - t**2 * lambdas**2)**2)
    return 2 * (f_t - 2 * k) * df_t

def grid_search(lambdas, k):
    """Perform a grid search over 10 points between 0 and 1/max(lambdas)."""
    t_max = 1 / np.max(lambdas)
    grid_points = np.linspace(1e-10, t_max - 1e-10, 10)  # 10 points between 0 and 1/max(lambdas)
    
    # Compute the objective function at each grid point
    objective_values = np.array([objective_function(t, lambdas, k) for t in grid_points])
    
    # Find the t that minimizes the objective function
    best_t_idx = np.argmin(objective_values)
    return grid_points[best_t_idx], objective_values[best_t_idx]

def optimize_t(lambdas, k, lr=1e-3, num_iters=1000):
    """Optimizes t using stochastic gradient descent, starting with the best t from grid search."""
    # Perform grid search for the initial guess of t
    t_init, _ = grid_search(lambdas, k)
    
    t = t_init  # Start optimization with the best t from grid search

    for _ in range(num_iters):
        grad = gradient(t, lambdas, k)
        t -= lr * grad

        # Ensure t is within bounds
        t = np.clip(t, 1e-8, 1 / np.max(lambdas) - 1e-8)

    return t, objective_function(t, lambdas, k)

def simple_grid_search(lambdas, k, n=int(1e4)):
    # Initial search range
    t_min, t_max = 0, 1 / np.max(lambdas)
    grid_points = np.linspace(t_min, t_max, n+2)
    grid_points = trim_vector(grid_points)
    with np.errstate(divide='ignore', invalid='ignore'):
        objective_values = [objective_function(t, lambdas, k) for t in grid_points]
    t_1_idx = np.argmin(objective_values)
    t_1 = grid_points[t_1_idx]
    min_obj_value = objective_values[t_1_idx]
    return t_1, min_obj_value

def adaptive_grid_search(lambdas, k, n=10, tol=1e-1):
    """Hybrid optimization algorithm using grid search and gradient-based refinement."""
    # Initial search range
    t_min, t_max = 0, 1 / np.max(lambdas)
    
    while True:
        # Step 1: Perform a grid search over the interval (t_min, t_max)
        grid_points = np.linspace(t_min, t_max, n+2)
        delta_t = grid_points[1] - grid_points[0]
        grid_points = trim_vector(grid_points)
        objective_values = [objective_function(t, lambdas, k) for t in grid_points]
        
        # Find t_1 that minimizes the objective function
        t_1_idx = np.argmin(objective_values)
        t_1 = grid_points[t_1_idx]
        min_obj_value = objective_values[t_1_idx]
        
        # Check stopping criterion
        if min_obj_value < tol:
            print("Converged with objective value:", min_obj_value)
            break
        
        # Step 2: Compute the gradient at t_1
        grad = gradient(t_1, lambdas, k)
        
        # Step 3: Select the new interval based on the gradient's sign
        if grad < 0:
            # Move to the left: (t_1 - delta_t, t_1]
            t_min = max(0, t_1 - delta_t)
            t_max = t_1
        else:
            # Move to the right: [t_1, t_1 + delta_t)
            t_min = t_1
            t_max = min(1 / np.max(lambdas), t_1 + delta_t)
            t_max = min(1, t_max)
    
    return t_1, min_obj_value

def trim_vector(vector):
    """Remove the first entry if it's 0 and the last entry if it's 1."""
    # Remove the first element if it's 0
    if vector[0] == 0:
        vector = vector[1:]
    
    # Remove the last element if it's 1
    if vector[-1] == 1:  # Check if the vector is non-empty
        vector = vector[:-1]
    
    return vector