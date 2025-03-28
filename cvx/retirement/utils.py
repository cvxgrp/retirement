import cvxpy as cp
import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm import trange
from .black_scholes import get_collar
from sklearn.mixture import GaussianMixture
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from .data_loader import load_data

def get_stationary_cov(A, Sigma_eps):
    """
    Returns \Sigma solving \Sigma = A \Sigma A^T + \Sigma^{\epsilon},
    """
    n = A.shape[0]

    Sigma = cp.Variable((n, n), PSD=True)
    objective = cp.Minimize(cp.norm(Sigma - A @ Sigma @ A.T - Sigma_eps, 'fro'))
    problem = cp.Problem(objective)
    problem.solve()

    return Sigma.value


def predict_inf_treas(x0, mu, A, n_steps):
    """
    param x0: initial values for inflation and treasury rate
    """

    predictions = pd.DataFrame(columns=['Inflation', 'Treasury'], index=range(n_steps), dtype=float)
    predictions.loc[0] = x0

    for time in range(1, n_steps+1):
        predictions.loc[time] = mu + A @ (predictions.loc[time-1] - mu)
        
    return predictions.iloc[1:]


def transform_data(data, k, s1, s2):
    """
    param k: kink point
    param s1: slope 1
    param s2: slope 2

    return: PWL transformed data
        s1 * (x - k) + k if x < k
        s2 * (x - k) + k if x >= k
    """
    def transform(x):
        if x < k:
            return s1 * (x - k) + k
        else:
            return s2 * (x - k) + k
        
    if isinstance(data, pd.Series):
        return pd.Series(np.array([transform(x) for x in data]), index=data.index, name=data.name)
    elif isinstance(data, np.ndarray):
        return np.array([transform(x) for x in data])
    elif isinstance(data, (int, float, np.int64, np.float64)):
        return transform(data)

def inverse_transform_data(data, k, s1, s2):
    """
    param data: PWL transformed data
    param k: kink point
    param s1: slope 1
    param s2: slope 2

    return: original data
        (x - k) / s1 + k if x < k
        (x - k) / s2 + k if x >= k
    """
    def inverse_transform(x):
        if x < k:
            return (x - k) / s1 + k
        else:
            return (x - k) / s2 + k
    
    if isinstance(data, pd.Series):
        return pd.Series(np.array([inverse_transform(x) for x in data]), index=data.index, name=data.name)
    elif isinstance(data, np.ndarray):
        return np.array([inverse_transform(x) for x in data])
    elif isinstance(data, (int, float, np.int64, np.float64)):
        return inverse_transform(data)

def solve_ar(X):
    """
    param X: time series data, each row is a sample

    returns: AR coefficients
    """

    a0 = np.mean(X, axis=0) # mean
    X_c = X - a0 # centered data

    X_plus = X_c[1:].T # X_t
    X_minus = X_c[:-1].T # X_{t-1}

    return a0, X_plus @ X_minus.T @ np.linalg.inv(X_minus @ X_minus.T)


def mean_reverting_walk(mu, sigma, n, theta=0.1):

    X = np.zeros(n)
    X[0] = mu
    epsilon = np.random.normal(0, sigma, n)
    for t in range(1, n):
        X[t] = X[t-1] + theta * (mu - X[t-1]) + epsilon[t]
    return X

def mean_reverting_walks(mu1, sigma1, mu2, sigma2, rho, n):
    """
    param mu1: mean of the first walk
    param sigma1: volatility of the first walk
    param mu2: mean of the second walk
    param sigma2: volatility of the second walk
    param rho: correlation between the two walks
    param n: number of steps
    """

    X = np.zeros(n)
    Y = np.zeros(n)
    X[0] = mu1
    Y[0] = mu2
    epsilon = np.random.normal(0, 1, n)
    for t in range(1, n):
        X[t] = X[t-1] + (mu1 - X[t-1]) + sigma1 * epsilon[t]
        Y[t] = Y[t-1] + (mu2 - Y[t-1]) + sigma2 * (rho * epsilon[t] + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1))
    return X, Y

def gmm_sample(gmm, component=None, n_dim=1):
    """
    Sample from a Gaussian Mixture Model (GMM).

    Parameters:
    - gmm: object with attributes `n_components`, `weights_`, `means_`, and `covariances_`.
    - component: Optional[int] - Specify which component to sample from. If None, randomly choose based on weights.
    - n_dim: int - Dimensionality of the sample. Assumes n_dim=1 corresponds to 1D Gaussians.

    Returns:
    - A sample (scalar for n_dim=1 or array for n_dim>1) from the GMM.
    """

    # Validate input
    if n_dim < 1:
        raise ValueError("n_dim must be at least 1.")
    if component is None:
        component = np.random.choice(gmm.n_components, p=gmm.weights_)
    elif not (0 <= component < gmm.n_components):
        raise ValueError(f"Component index out of range: {component}")

    if n_dim == 1:
        # 1D case
        mean = gmm.means_[component]
        variance = gmm.covariances_[component]  # Ensure this is a scalar
        return np.random.normal(mean, np.sqrt(variance)).flatten()[0]
    else:
        # Multivariate case
        mean = gmm.means_[component]
        cov = gmm.covariances_[component]
        if mean.shape[0] != n_dim or cov.shape != (n_dim, n_dim):
            raise ValueError("Mean or covariance dimension mismatch with n_dim.")
        return np.random.multivariate_normal(mean, cov)
