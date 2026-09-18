import torch

def kv_compress(x: torch.Tensor, W_dkv: torch.Tensor) -> torch.Tensor:
    return x@W_dkv.T
    """
    Returns compressed KV latents of shape (batch, sequence, latent_width).
    """
   
    pass