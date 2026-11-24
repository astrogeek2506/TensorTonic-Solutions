
import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """
    Returns the float64 reconstruction with shape (B, D).
    """
    z = np.asarray(z, dtype=np.float64)
    W_dec = np.asarray(W_dec, dtype=np.float64)
    b_dec = np.asarray(b_dec, dtype=np.float64)

    logits = z @ W_dec + b_dec

    x_hat = 1.0 / (1.0 + np.exp(-logits))

    return x_hat.astype(np.float64)
