"""Utility functions for loss computation."""

import torch
import numpy as np


def parse_multi_negative_samples(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_hard_negatives: int = None,
) -> list:
    """
    Parse a flat tensor of embeddings into per-query groups.

    Each group is structured as: [anchor, positive, neg_1, neg_2, ..., neg_n]

    The `labels` tensor uses 1 to mark the start of each sample (the anchor position),
    and 0 for all other positions (positive + negatives).

    Args:
        embeddings: [N, D] tensor of all embeddings in the batch.
        labels: [N] tensor where 1 indicates the start of a new sample.
        max_hard_negatives: If set, truncate or pad negatives to this number.

    Returns:
        List of tensors, each of shape [1 + 1 + num_negatives, D].

    Example:
        For batch_size=2 with 3 and 2 negatives respectively:
        labels = [1, 0, 0, 0, 1, 0, 0]
        Returns: [tensor([anchor1, pos1, neg1, neg2, neg3]),
                  tensor([anchor2, pos2, neg1, neg2])]
    """
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]

    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))

    split_tensors = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = embeddings[start:end]

        if max_hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0, "Each sample must have at least one negative"
            if negatives > max_hard_negatives:
                split_part = split_part[:max_hard_negatives + 2]
            elif negatives < max_hard_negatives:
                selected = np.random.choice(
                    list(range(negatives)),
                    size=max_hard_negatives - negatives,
                    replace=True
                )
                selected += 1
                split_part = torch.cat((split_part, split_part[selected]), dim=0)

        split_tensors.append(split_part)

    return split_tensors
