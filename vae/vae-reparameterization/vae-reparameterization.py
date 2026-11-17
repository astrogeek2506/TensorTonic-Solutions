
import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns the float64 latent sample with the same shape as mu.
    """
    mu = np.asarray(mu, dtype=np.float64)
    log_var = np.asarray(log_var, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)

    std = np.exp(0.5 * log_var)

    z = mu + std * epsilon

    return z.astype(np.float64)
