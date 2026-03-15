import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs)
    """

    N = len(seqs)

    if max_len is None:
        L = max((len(seq) for seq in seqs), default=0)
    else:
        L = max_len

    # create padded matrix
    padded = np.full((N, L), pad_value, dtype=int)

    # fill sequences
    for i, seq in enumerate(seqs):
        trunc = seq[:L]   # truncate if needed
        padded[i, :len(trunc)] = trunc

    return padded