
import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray,
                W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns mu and log_var as float64 arrays in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    W_mu = np.asarray(W_mu, dtype=np.float64)
    b_mu = np.asarray(b_mu, dtype=np.float64)
    W_logvar = np.asarray(W_logvar, dtype=np.float64)
    b_logvar = np.asarray(b_logvar, dtype=np.float64)

    mu = x @ W_mu + b_mu
    log_var = x @ W_logvar + b_logvar

    return {
        "mu": mu,
        "log_var": log_var
    }
