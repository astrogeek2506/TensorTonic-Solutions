import torch
from typing import Optional
import numpy as np
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    dk=query.shape[-1]
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(dk)

    if mask is not None:
        scores=scores.masked_fill(mask,float("-inf"))

    weights=torch.softmax(scores,dim=-1)
    output=torch.matmul(weights,value)
    



    
    return output
