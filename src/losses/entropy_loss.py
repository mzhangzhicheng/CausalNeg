"""
Query-Perspective Entropy Maximization Loss.

This module implements our proposed auxiliary loss that prevents the model from
learning shortcut features to distinguish generated negatives from mined negatives.

Core Idea:
    From the query's perspective, the similarity distribution over all negatives 
    (mined + generated) should not reveal the source of each negative. We achieve 
    this by maximizing the entropy of generated negatives' probability distribution 
    and balancing their total probability mass.

Loss Formulation:
    L_total = L_InfoNCE + lambda * L_entropy
    
    where L_entropy = -H_gen + L_balance
    
    H_gen: Entropy of generated negatives' normalized probability distribution
    L_balance: Squared difference between actual and expected probability mass
    
    Only the generated negatives receive gradients from L_entropy, leaving 
    the mined negatives' learning unaffected.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .infonce import infonce_loss_batched
from .utils import parse_multi_negative_samples

logger = logging.getLogger(__name__)


def query_entropy_loss(
    query: torch.Tensor,
    mined_negatives: torch.Tensor,
    generated_negatives: torch.Tensor,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute query-perspective entropy maximization loss for a single sample.
    
    For a given query q, we compute the similarity distribution over all negatives
    (mined ∪ generated) and encourage the generated negatives to be indistinguishable 
    from mined negatives.
    
    The loss has two components:
    1. **Entropy Maximization**: Maximize the entropy of the generated negatives' 
       normalized probability distribution, encouraging uniform probability among 
       generated negatives.
    2. **Probability Balance**: Penalize deviation of the generated negatives' total 
       probability from the expected uniform share.
    
    Args:
        query: [D] query embedding (detached, no gradient).
        mined_negatives: [N_m, D] mined negative embeddings (detached, no gradient).
        generated_negatives: [N_g, D] generated negative embeddings (with gradient).
        temperature: Temperature for softmax computation (tau_ent).
    
    Returns:
        loss: Scalar entropy loss for this sample.
        stats: Dictionary with monitoring statistics.
    
    Mathematical Formulation:
        s_i = sim(q, d_i^-) / tau_ent    for d_i^- in N_mined ∪ N_gen
        p_i = softmax(s)_i
        
        # Normalized probabilities within generated negatives
        p_gen_norm_i = p_i / sum(p_j for j in N_gen)
        
        # Entropy of generated negative distribution
        H_gen = -sum(p_gen_norm_i * log(p_gen_norm_i))
        
        # Probability balance constraint
        L_balance = (sum(p_i for i in N_gen) - |N_gen| / (|N_mined| + |N_gen|))^2
        
        # Final entropy loss
        L_entropy = -H_gen + L_balance
    """
    num_mined = mined_negatives.size(0)
    num_gen = generated_negatives.size(0)
    
    # Detach query and mined negatives - only update generated negatives
    query_detached = query.detach()
    mined_detached = mined_negatives.detach()
    
    # Concatenate all negatives: [N_m + N_g, D]
    all_negatives = torch.cat([mined_detached, generated_negatives], dim=0)
    
    # Compute query-to-negative similarities: [N_m + N_g]
    similarities = torch.matmul(query_detached, all_negatives.T) / temperature
    
    # Softmax probabilities over all negatives
    probabilities = F.softmax(similarities, dim=-1)
    
    # Extract generated negatives' probabilities
    gen_probs = probabilities[num_mined:]  # [N_g]
    
    # --- Component 1: Entropy Maximization ---
    # Normalize generated probabilities to form a valid distribution
    gen_probs_normalized = gen_probs / (gen_probs.sum() + 1e-10)
    log_gen_probs = torch.log(gen_probs_normalized + 1e-10)
    gen_entropy = -torch.sum(gen_probs_normalized * log_gen_probs)
    
    # --- Component 2: Probability Balance ---
    expected_gen_prob = num_gen / (num_mined + num_gen)
    actual_gen_prob = gen_probs.sum()
    balance_loss = (actual_gen_prob - expected_gen_prob) ** 2
    
    # --- Combined Entropy Loss ---
    loss = -gen_entropy + balance_loss
    
    stats = {
        'gen_entropy': gen_entropy.item(),
        'actual_gen_prob': actual_gen_prob.item(),
        'expected_gen_prob': expected_gen_prob,
        'balance_loss': balance_loss.item(),
        'num_gen': num_gen,
    }
    
    return loss, stats


def infonce_with_entropy_loss(
    split_tensors: list,
    temperature: float = 0.05,
    entropy_temperature: float = 0.1,
    entropy_weight: float = 0.05,
    num_mined_negatives: int = 15,
    use_in_batch_negatives: bool = False,
    world_size: int = 1,
    log_interval: int = 10,
    step_counter: Optional[list] = None,
) -> torch.Tensor:
    """
    InfoNCE + Query-Perspective Entropy Maximization combined loss.
    
    This is the main loss function of our method. It combines the standard InfoNCE 
    contrastive loss with our proposed entropy maximization auxiliary loss.
    
    Data Format:
        Each tensor in split_tensors has shape [1+1+N_m+N_g, D]:
        - Index 0: query (anchor) embedding
        - Index 1: positive document embedding
        - Index 2 to 2+N_m: mined negative embeddings
        - Index 2+N_m to end: generated negative embeddings
    
    The number of generated negatives is dynamically computed per sample:
        N_g = tensor.size(0) - 2 - N_m
    
    Args:
        split_tensors: List of per-sample embedding tensors.
        temperature: Temperature for InfoNCE loss (tau).
        entropy_temperature: Temperature for entropy loss (tau_ent).
        entropy_weight: Weight lambda for entropy loss.
        num_mined_negatives: Number of mined negatives per sample (N_m).
        use_in_batch_negatives: Whether to use cross-batch negatives for InfoNCE.
        world_size: Number of distributed processes.
        log_interval: Steps between detailed log outputs.
        step_counter: Mutable list [count] for tracking step number.
    
    Returns:
        total_loss: L_InfoNCE + lambda * L_entropy
    """
    device = split_tensors[0].device
    D = split_tensors[0].size(1)
    B = len(split_tensors)
    
    # ============================================================
    # Part 1: Standard InfoNCE Loss
    # ============================================================
    main_loss = infonce_loss_batched(
        split_tensors, 
        temperature=temperature,
        use_in_batch_negatives=use_in_batch_negatives,
        world_size=world_size,
    )
    
    # ============================================================
    # Part 2: Query-Perspective Entropy Maximization Loss
    # ============================================================
    entropy_losses = []
    all_stats = []
    
    # Collect embeddings for logging
    all_gen_embs = []
    all_mined_embs = []
    all_query_embs = []
    
    for tensor in split_tensors:
        sample_len = tensor.size(0)
        num_gen_actual = sample_len - 2 - num_mined_negatives
        
        if num_gen_actual <= 0:
            # No generated negatives in this sample, skip entropy loss
            continue
        
        query = tensor[0]  # [D]
        mined = tensor[2:2 + num_mined_negatives]  # [N_m, D]
        gen = tensor[2 + num_mined_negatives:2 + num_mined_negatives + num_gen_actual]  # [N_g, D]
        
        loss, stats = query_entropy_loss(
            query=query,
            mined_negatives=mined,
            generated_negatives=gen,
            temperature=entropy_temperature,
        )
        
        entropy_losses.append(loss)
        all_stats.append(stats)
        
        all_gen_embs.append(gen)
        all_mined_embs.append(mined)
        all_query_embs.append(query)
    
    # Average entropy loss across valid samples
    if len(entropy_losses) > 0:
        entropy_loss = torch.stack(entropy_losses).mean()
    else:
        entropy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # ============================================================
    # Logging
    # ============================================================
    if step_counter is None:
        step_counter = [0]
    step_counter[0] += 1
    
    if step_counter[0] % log_interval == 0 and len(all_stats) > 0:
        _log_training_stats(
            step=step_counter[0],
            main_loss=main_loss,
            entropy_loss=entropy_loss,
            entropy_weight=entropy_weight,
            stats=all_stats,
            gen_embs=all_gen_embs,
            mined_embs=all_mined_embs,
            query_embs=all_query_embs,
            B=B,
            num_mined=num_mined_negatives,
        )
    
    # ============================================================
    # Combined Loss
    # ============================================================
    total_loss = main_loss + entropy_weight * entropy_loss
    return total_loss


def _log_training_stats(
    step, main_loss, entropy_loss, entropy_weight, stats,
    gen_embs, mined_embs, query_embs, B, num_mined,
):
    """Log detailed training statistics for monitoring."""
    with torch.no_grad():
        B_valid = len(gen_embs)
        mean_gen_entropy = np.mean([s['gen_entropy'] for s in stats])
        mean_actual_gen_prob = np.mean([s['actual_gen_prob'] for s in stats])
        mean_num_gen = np.mean([s['num_gen'] for s in stats])
        max_gen_entropy = np.log(mean_num_gen) if mean_num_gen > 1 else 1.0
        
        # Embedding similarity metrics
        all_gen_flat = torch.cat(gen_embs, dim=0)
        all_mined_flat = torch.cat(mined_embs, dim=0)
        total_gen = all_gen_flat.size(0)
        total_mined = all_mined_flat.size(0)
        
        gen_mined_sim = (all_gen_flat @ all_mined_flat.T).mean().item()
        
        if total_gen > 1:
            gen_gen_sim_matrix = all_gen_flat @ all_gen_flat.T
            gen_gen_sim = (gen_gen_sim_matrix.sum() - gen_gen_sim_matrix.trace()) / (total_gen * (total_gen - 1))
        else:
            gen_gen_sim = torch.tensor(0.0)
        
        if total_mined > 1:
            mined_mined_sim_matrix = all_mined_flat @ all_mined_flat.T
            mined_mined_sim = (mined_mined_sim_matrix.sum() - mined_mined_sim_matrix.trace()) / (total_mined * (total_mined - 1))
        else:
            mined_mined_sim = torch.tensor(0.0)
        
        # Query-based similarity metrics
        sim_q_mined = np.mean([(q @ m.T).mean().item() for q, m in zip(query_embs, mined_embs)])
        sim_q_gen = np.mean([(q @ g.T).mean().item() for q, g in zip(query_embs, gen_embs)])
        
        gen_entropy_ratio = mean_gen_entropy / max_gen_entropy if max_gen_entropy > 0 else 0
        expected_gen_prob = mean_num_gen / (num_mined + mean_num_gen) if mean_num_gen > 0 else 0
        
        total = main_loss.item() + entropy_weight * entropy_loss.item()
        logger.info(
            f"[Entropy] step={step} | "
            f"main={main_loss.item():.4f}, ent_loss={entropy_loss.item():.4f}, total={total:.4f} | "
            f"valid={B_valid}/{B}, avg_gen={mean_num_gen:.1f} | "
            f"gen_entropy={mean_gen_entropy:.3f}/{max_gen_entropy:.3f} (ratio={gen_entropy_ratio:.3f}) | "
            f"query_sim: q-mined={sim_q_mined:.3f}, q-gen={sim_q_gen:.3f} | "
            f"neg_prob: gen={mean_actual_gen_prob:.4f} (expect={expected_gen_prob:.4f}) | "
            f"emb_sim: g-m={gen_mined_sim:.3f}, g-g={gen_gen_sim.item():.3f}, m-m={mined_mined_sim.item():.3f}"
        )
