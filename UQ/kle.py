import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
from scipy.optimize import fsolve
from scipy.linalg import eigh


def compute_kle_parameters(a, l, d):
    """
    Analytically compute parameters related to the Karhunen-Loève Expansion (KLE) for 
    a Gaussian process with an exponential covariance function.

    Parameters:
    - a: float, spatial domain boundary.
    - l: float, correlation length for the exponential covariance function.
    - d: int, number of terms in the KLE truncation.

    Returns:
    - KLEIG: numpy.ndarray, an array storing the parameters (omega, lambda, coefficient)
             for each term in the KLE truncation.
    """
    # Initialize the KLEIG matrix to store the parameters
    KLEIG = np.zeros((d + 1, 3))

    for i in range(int(np.ceil(d / 2)) + 1):
        # Define the interval for finding zeros based on the index i
        intv = [max((2 * i - 1) * np.pi / (2 * a) + 1e-8, 0), (2 * i + 1) * np.pi / (2 * a) - 1e-8]
        
        # Handle even terms
        if i > 0 and 2 * i <= d:
            w = fsolve(lambda x: (1 / l) * np.tan(a * x) + x, np.mean(intv))[0]
            # Check for very small w to avoid division by zero or very small numbers
            if np.abs(w) < 1e-8:
                print(f"w is too small: {w}, setting sqrt_arg to a default positive value.")
                sqrt_arg = a  # Setting sqrt_arg to a default positive value to avoid negative sqrt argument
            else:
                sqrt_arg = a - np.sin(2 * w * a) / (2 * w)
            
            # Ensure sqrt_arg is non-negative before taking the square root
            if sqrt_arg < 0:
                print(f"sqrt_arg is negative: {sqrt_arg}, using its absolute value to avoid complex numbers.")
                sqrt_arg = np.abs(sqrt_arg)
            
            KLEIG[2 * i - 1, :] = [w, 2 * l / (w ** 2 * l ** 2 + 1), 1 / np.sqrt(sqrt_arg)]
        
        # Handle odd terms
        if 2 * i + 1 <= d:
            w = fsolve(lambda x: (1 / l) - x * np.tan(a * x), np.mean(intv))[0]
            # Check for very small w to avoid division by zero or very small numbers
            if np.abs(w) < 1e-8:
                print(f"w is too small: {w}, setting sqrt_arg to a default positive value.")
                sqrt_arg = a  # Setting sqrt_arg to a default positive value to avoid negative sqrt argument
            else:
                sqrt_arg = a + np.sin(2 * w * a) / (2 * w)
            
            # Ensure sqrt_arg is non-negative before taking the square root
            if sqrt_arg < 0:
                print(f"sqrt_arg is negative: {sqrt_arg}, using its absolute value to avoid complex numbers.")
                sqrt_arg = np.abs(sqrt_arg)

            KLEIG[2 * i, :] = [w, 2 * l / (w ** 2 * l ** 2 + 1), 1 / np.sqrt(sqrt_arg)]
            
    return KLEIG


def generate_gaussian_process_realizations(n_samples, t, d, KLEIG, sigma, exact_mean):
    """
    Generate realizations of a Gaussian process using the Karhunen-Loève Expansion.

    Parameters:
    - n_samples: int, the number of samples to generate.
    - t: numpy.ndarray, the spatial domain over which to generate the process.
    - d: int, the number of terms in the KLE truncation.
    - KLEIG: numpy.ndarray, a matrix storing the parameters (omega, lambda, coefficient) for each KLE term.
    - sigma: float, the standard deviation of the process.
    - exact_mean: float, the mean value to be added to each realization.

    Returns:
    - realizations: numpy.ndarray, an array of generated realizations of the Gaussian process.
    """
    # Initialize an array to store the realizations
    realizations = np.zeros((n_samples, len(t)))
    
    # Generate each realization by adding contributions from the KLE
    for n in range(n_samples):  # For each sample
        for i in range(int(np.ceil(d / 2)) + 1):  # For each term in the KLE
            # Add contributions from even terms
            if i > 0 and 2 * i <= d:
                realizations[n, :] += sigma * np.sqrt(KLEIG[2 * i - 1, 1]) * KLEIG[2 * i - 1, 2] * np.sin(KLEIG[2 * i - 1, 0] * t) * npr.randn()
            # Add contributions from odd terms
            if 2 * i + 1 <= d:
                realizations[n, :] += sigma * np.sqrt(KLEIG[2 * i, 1]) * KLEIG[2 * i, 2] * np.cos(KLEIG[2 * i, 0] * t) * npr.randn()
        # Add the exact mean to each realization
        realizations[n, :] += exact_mean

    return realizations


def G_mean():
    """
    Define the constant mean of the Gaussian process G(x, w).
    
    Returns:
    float: The mean value of the Gaussian process, set to 1.0.
    """
    return 1.0


def C_GG(x1, x2, sigma, l):
    """
    Define the covariance function of the Gaussian process G(x, w).

    Parameters:
    - x1: float, the first spatial point.
    - x2: float, the second spatial point.
    - sigma: float, the standard deviation of the process, controlling the overall 
    variance.
    - l: float, the correlation length, determining how quickly the correlation 
    between points decreases with distance.

    Returns:
    float: The value of the covariance function between two points x1 and x2.
    """
    return sigma ** 2 * np.exp(-np.abs(x1 - x2) / l)


def compute_eigenpairs(sigma, l, grid_points):
    """
    Compute the eigenvalues and eigenvectors (eigenpairs) of the covariance matrix 
    constructed from the covariance function C_GG over a discretized spatial domain.

    Parameters:
    - sigma: float, the standard deviation of the Gaussian process.
    - l: float, the correlation length of the Gaussian process.
    - grid_points: int, the number of points in the discretized spatial domain.

    Returns:
    tuple: Contains three elements:
        - x: numpy.ndarray, the discretized spatial domain.
        - eigvals: numpy.ndarray, the eigenvalues of the covariance matrix, 
        sorted in descending order.
        - eigvecs: numpy.ndarray, the eigenvectors of the covariance matrix, 
        corresponding to the sorted eigenvalues.
    """
    x = np.linspace(0, 1, grid_points)
    cov_matrix = np.array([[C_GG(x1, x2, sigma, l) for x2 in x] for x1 in x])
    eigvals, eigvecs = eigh(cov_matrix)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    return x, eigvals, eigvecs


def n_terms_for_kl_expansion(eigvals, target):
    """
    Determine the number of terms needed in the Karhunen-Loève expansion to 
    achieve a given truncation error target.

    Parameters:
    - eigvals: numpy.ndarray, the eigenvalues of the covariance matrix.
    - target: float, the target proportion of total variance to capture 
    (e.g., 0.9 to capture 90% of the variance).

    Returns:
    int: The minimum number of terms required to achieve the target 
    truncation error.
    """
    total_var = np.sum(eigvals)
    cumulative_sum = np.cumsum(eigvals)
    n_terms = np.argmax(cumulative_sum >= (1 - target) * total_var) + 1
    if n_terms == 1 and cumulative_sum[0] < (1 - target) * total_var:
        n_terms = len(eigvals)
    return n_terms


def gen_realizations(x, eigvecs, eigvals, n_realizations, n_terms):
    """
    Generate realizations of the Gaussian process G(x, w) using the 
    Karhunen-Loève expansion.

    Parameters:
    - x: numpy.ndarray, the discretized spatial domain.
    - eigvecs: numpy.ndarray, the eigenvectors of the covariance matrix.
    - eigvals: numpy.ndarray, the eigenvalues of the covariance matrix.
    - n_realizations: int, the number of realizations to generate.
    - n_terms: int, the number of terms to use in the KL expansion.

    Returns:
    numpy.ndarray: An array of generated realizations of the Gaussian process.
    """
    realizations = []
    for _ in range(n_realizations):
        coefs = np.random.randn(n_terms)
        realization = np.sum(np.sqrt(eigvals[:n_terms]) * eigvecs[:, :n_terms] * coefs, axis=1)
        realization += G_mean()  # Add the mean function to each realization
        realizations.append(realization)
    return np.array(realizations)


def plot_sample_mean_var(x, sigma, mean, sample_means, sample_variances):
    # plot the sample mean and variance against the exact mean and variance
    plt.figure(figsize=(10, 6))

    # plot the sample mean
    plt.subplot(1, 2, 1)
    plt.plot(x, sample_means, label='Sample Mean of $G(x,\\omega)$')
    plt.axhline(y=exact_mean, color='r', linestyle='--', label='Exact Mean = 1')
    plt.title('Sample Mean and Exact Mean of $G(x,\\omega)$')
    plt.xlabel('x')
    plt.ylabel('Mean of $G(x,\\omega)$')
    plt.legend()

    # plot the sample variance
    plt.subplot(1, 2, 2)
    plt.plot(x, sample_variances, label='Sample Variance of $G(x,\\omega)$')
    plt.axhline(y=sigma**2, color='r', linestyle='--', label=f'Exact Variance = {sigma**2}')
    plt.title('Sample Variance and Exact Variance of $G(x,\\omega)$')
    plt.xlabel('x')
    plt.ylabel('Variance of $G(x,\\omega)$')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test_KLEIG(a, l, d, exact_mean, sigma, n_samples, t):

    KLEIG = compute_kle_parameters(a, l, d)
    realizations = generate_gaussian_process_realizations(n_samples, t, d, KLEIG, sigma, exact_mean)

    # Compute the sample mean and variance from the realizations
    sample_mean = np.mean(realizations, axis=0)  # Compute the mean across all samples for each spatial location
    sample_variance = np.var(realizations, axis=0)  # Compute the variance across all samples for each spatial location

    plot_sample_mean_var(t, sigma, exact_mean, sample_mean, sample_variance)



def test_numerical_KLE(l, sigma, n_samples, grid_points, target):

    # compute eigen-pairs
    x, eigvals, eigvecs = compute_eigenpairs(sigma, l, grid_points)
    # determine the number of terms required for the KL expansion
    n_terms = n_terms_for_kl_expansion(eigvals, target)
    print(f'number of terms for KL expansion: {n_terms}: {eigvals[:n_terms]}')

    # draw samples from the KL expansion
    sample_realizations = gen_realizations(x, eigvecs, eigvals, n_samples, n_terms)
    # compute the sample mean and variance as a function of x
    sample_means = np.mean(sample_realizations, axis=0)
    sample_variances = np.var(sample_realizations, axis=0)

    plot_sample_mean_var(x, sigma, exact_mean, sample_means, sample_variances)




if __name__ == "__main__":

    # Define constants and parameters for the simulation
    l = 0.2  # Correlation length for the exponential covariance function
    exact_mean = G_mean()  # The exact mean of the process, to be added to each realization
    sigma = 2  # Standard deviation of the process
    n_samples = 100000  # Number of samples to generate

    # additional paramters for analytical approach
    a = 1  # Spatial domain boundary
    d = 2  # Number of terms in the Karhunen-Loève Expansion (KLE) truncation
    t = np.arange(-1*a, a + 0.005, 0.005)  # Spatial domain from -a to a, with small steps

    # additional paramters for numerical approach
    grid_points = 100 # number of grid points for discretization
    target = 0.10 # 10% target mean-square error of truncation

    test_KLEIG(a, l, d, exact_mean, sigma, n_samples, t)

    test_numerical_KLE(l, sigma, n_samples, grid_points, target)
    