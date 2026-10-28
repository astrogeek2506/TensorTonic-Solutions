
import torch

def match_prefix_cache(
    request_token_ids: list,
    cached_token_blocks: list,
    cached_physical_block_ids: list,
    block_size: int,
) -> tuple:

    num_request_blocks = len(request_token_ids) // block_size

    best_match = 0
    best_candidate_id = None

    for i, candidate_blocks in enumerate(cached_token_blocks):
        matched = 0

        limit = min(num_request_blocks, len(candidate_blocks))

        for k in range(limit):
            start = k * block_size
            end = start + block_size

            request_block = request_token_ids[start:end]
            cached_block = candidate_blocks[k]

            if list(request_block) == list(cached_block):
                matched += 1
            else:
                break

        # Update the winner only after checking this candidate.
        if matched > best_match:
            best_match = matched
            best_candidate_id = i

    if best_candidate_id is None:
        return (
            torch.tensor(0, dtype=torch.long),
            torch.tensor([], dtype=torch.long),
        )

    reusable_ids = cached_physical_block_ids[best_candidate_id][:best_match]
    matched_token_count = best_match * block_size

    return (
        torch.tensor(matched_token_count, dtype=torch.long),
        torch.tensor(reusable_ids, dtype=torch.long),
    )
