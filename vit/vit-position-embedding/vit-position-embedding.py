import numpy as np

def add_position_embedding(
    patches: np.ndarray,
    num_patches: int,
    embed_dim: int,
    pos_embed: np.ndarray = None
) -> np.ndarray:
    """
    Add position embeddings to patch embeddings.

    Args:
        patches: Patch embeddings of shape (B, N, D)
        num_patches: Number of tokens N
        embed_dim: Embedding dimension D
        pos_embed: Position embeddings of shape (1, N, D).
                   If None, initialize randomly.

    Returns:
        Position-enhanced embeddings of shape (B, N, D)
    """
    if pos_embed is None:
        pos_embed = np.random.randn(1, num_patches, embed_dim) * 0.02

    # Validate shape
    assert pos_embed.shape == (1, num_patches, embed_dim), \
        f"Expected pos_embed shape (1, {num_patches}, {embed_dim}), got {pos_embed.shape}"

    # Broadcasting adds the same positional embedding to every sample in the batch
    return patches + pos_embed