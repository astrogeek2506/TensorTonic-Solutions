def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        chunks.append(chunk)

        # stop once we reach the end
        if start + chunk_size >= len(tokens):
            break

    return chunks