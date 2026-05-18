"""
Standard InfoNCE Loss for Contrastive Learning.

This module implements the InfoNCE loss function used as the baseline
in our experiments. It supports both in-batch negatives and per-sample
negatives with variable-length negative sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import parse_multi_negative_samples


def infonce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Standard InfoNCE loss for a single query.

    L = -log( exp(sim(q, d+) / tau) / (exp(sim(q, d+) / tau) + sum_i exp(sim(q, d_i-) / tau)) )

    Args:
        query: [D] query embedding.
        positive: [D] positive document embedding.
        negatives: [N, D] negative document embeddings.
        temperature: Temperature scaling factor tau.

    Returns:
        Scalar loss value.
    """
    candidates = torch.cat([positive.unsqueeze(0), negatives], dim=0)
    similarities = torch.matmul(query, candidates.T) / temperature
    target = torch.tensor(0, device=query.device)
    return F.cross_entropy(similarities.unsqueeze(0), target.unsqueeze(0))


def infonce_loss_batched(
    split_tensors: list,
    temperature: float = 0.05,
    use_in_batch_negatives: bool = True,
    world_size: int = 1,
) -> torch.Tensor:
    """
    Batched InfoNCE loss supporting both in-batch and per-sample negatives.

    Each element in split_tensors is a tensor of shape [1 + 1 + num_neg, D],
    where the first row is the anchor (query), the second is the positive,
    and the rest are negatives.

    When use_in_batch_negatives=True, all positives and negatives across samples
    are used as candidates for each query (cross-batch negatives).

    Args:
        split_tensors: List of tensors, each [1+1+N_i, D].
        temperature: Temperature scaling factor.
        use_in_batch_negatives: Whether to use cross-batch negatives.
        world_size: Number of distributed processes (for loss normalization).

    Returns:
        Scalar loss value.
    """
    device = split_tensors[0].device
    B = len(split_tensors)

    if not use_in_batch_negatives:
        can_batched = len(set(t.shape[0] for t in split_tensors)) == 1

        if can_batched:
            sentences = torch.stack(split_tensors, dim=0)
            sim = torch.matmul(
                sentences[:, 0:1],
                sentences[:, 1:].transpose(1, 2)
            ) / temperature
            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(sim.squeeze(1), labels)
        else:
            loss = torch.tensor(0.0, device=device)
            for tensor in split_tensors:
                sim = torch.matmul(tensor[0], tensor[1:].T) / temperature
                target = torch.tensor(0, device=device)
                loss += F.cross_entropy(sim.unsqueeze(0), target.unsqueeze(0))
            loss /= B
    else:
        all_candidates = torch.cat([t[1:] for t in split_tensors], dim=0)

        loss = torch.tensor(0.0, device=device)
        length = 0
        for tensor in split_tensors:
            anchor = tensor[0]
            sim = torch.matmul(anchor, all_candidates.T) / temperature
            target = torch.tensor(length, device=device)
            loss += F.cross_entropy(sim.unsqueeze(0), target.unsqueeze(0))
            length += tensor.size(0) - 1

        loss = loss / B / world_size

    return loss
