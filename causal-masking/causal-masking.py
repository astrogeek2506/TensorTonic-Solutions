import numpy as np

def apply_causal_mask(scores, mask_value=-np.inf):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: -np.inf for strict masking
    """
    T = scores.shape[-1]
    
    # Create the boolean mask for upper triangular
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    
    # Ensure scores are float to handle infinity
    masked_scores = scores.astype(float)
    
    # Use the boolean mask to set future positions to -inf
    masked_scores[..., mask] = mask_value
    
    return masked_scores