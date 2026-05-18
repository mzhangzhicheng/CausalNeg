"""
MS-Swift Framework Integration Plugin.

This file provides the loss function implementation compatible with the
MS-Swift (https://github.com/modelscope/ms-swift) training framework.

To use this plugin:
1. Copy this file to: ms-swift/swift/plugin/loss.py (or add the functions)
2. Register in loss_mapping: 'infonce_entropy': infonce_entropy_loss
3. Set environment variables for hyperparameters
4. Use --loss_type infonce_entropy in training command

Environment Variables:
    INFONCE_TEMPERATURE: InfoNCE temperature (default: 0.05)
    INFONCE_USE_BATCH: Use cross-batch negatives (default: False)
    ENTROPY_TEMPERATURE: Entropy loss temperature (default: 0.1)
    ENTROPY_WEIGHT: Entropy loss weight lambda (default: 0.05)
    NUM_MINED_NEGATIVES: Number of mined negatives (default: 15)
    NUM_GENERATED_NEGATIVES: Number of generated negatives (default: 3)
    CONFUSION_LOG_INTERVAL: Logging interval (default: 10)
"""

import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import gather_object
from transformers.utils import strtobool

logger = logging.getLogger(__name__)


def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    """Parse flat embeddings into per-query groups based on labels."""
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []
    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]
            elif negatives < hard_negatives:
                selected = np.random.choice(
                    list(range(negatives)), size=hard_negatives - negatives, replace=True
                )
                selected += 1
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors


def infonce_entropy_loss(
    outputs, labels, loss_scale=None, num_items_in_batch=None, **kwargs
) -> torch.Tensor:
    """
    InfoNCE + Query-Perspective Entropy Maximization Loss.
    
    This loss function combines standard InfoNCE contrastive loss with our
    proposed entropy maximization auxiliary loss that prevents the model from
    learning shortcut features to distinguish generated vs mined negatives.
    
    The auxiliary loss operates from the query's perspective:
    1. Computes similarity distribution over all negatives (mined + generated)
    2. Maximizes entropy of generated negatives' probability distribution
    3. Balances total probability mass between mined and generated negatives
    4. Only backpropagates through generated negative embeddings
    
    Compatible with MS-Swift framework's loss plugin system.
    """
    # Main loss parameters
    temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.02'))
    use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))

    # Entropy loss parameters
    entropy_temp = float(os.environ.get('ENTROPY_TEMPERATURE', '0.1'))
    entropy_weight = float(os.environ.get('ENTROPY_WEIGHT', '0.1'))
    num_mined = int(os.environ.get('NUM_MINED_NEGATIVES', '15'))
    num_gen_config = int(os.environ.get('NUM_GENERATED_NEGATIVES', '3'))
    log_interval = int(os.environ.get('CONFUSION_LOG_INTERVAL', '10'))

    from swift.utils import get_dist_setting
    rank, _, world_size, _ = get_dist_setting()
    sentences = outputs['last_hidden_state']

    # Cross-GPU gather for distributed training
    if world_size > 1 and use_batch:
        all_sentences = gather_object(sentences.unsqueeze(0))
        labels = gather_object(labels)
        all_sentences[rank] = sentences
        for idx in range(len(all_sentences)):
            if idx != rank:
                all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
        sentences = torch.cat(all_sentences, dim=0)
        labels = [tensor.to(sentences.device) for tensor in labels]
        labels = torch.stack(labels, dim=0)

    # Parse into per-query groups
    split_tensors = _parse_multi_negative_sentences(sentences, labels, None)
    B = len(split_tensors)
    device = split_tensors[0].device
    D = split_tensors[0].size(1)

    # ========== Part 1: Standard InfoNCE Loss ==========
    all_candidates = []
    for tensor in split_tensors:
        all_candidates.append(tensor[1:])
    all_candidates_cat = torch.cat(all_candidates, dim=0)

    main_loss = 0.0
    length = 0
    for tensor in split_tensors:
        anchor = tensor[0]
        sim = torch.matmul(anchor, all_candidates_cat.T) / temperature
        target = torch.tensor(length, device=device)
        main_loss = main_loss + F.cross_entropy(sim.unsqueeze(0), target.unsqueeze(0))
        length += tensor.size(0) - 1
    main_loss = main_loss / len(split_tensors) / world_size

    # ========== Part 2: Query-Perspective Entropy Maximization ==========
    entropy_losses = []
    gen_entropies = []
    actual_gen_probs = []
    gen_counts = []
    all_gen_embs = []
    all_mined_embs = []
    all_query_embs = []

    for tensor in split_tensors:
        sample_len = tensor.size(0)
        num_gen_actual = sample_len - 2 - num_mined

        if num_gen_actual <= 0:
            continue

        query = tensor[0]
        mined = tensor[2:2 + num_mined]
        gen = tensor[2 + num_mined:2 + num_mined + num_gen_actual]

        # Detach query and mined - only update generated negatives
        mined_detached = mined.detach()
        query_detached = query.detach()

        # Concatenate all negatives
        all_negs = torch.cat([mined_detached, gen], dim=0)

        # Query-to-negative similarities
        sim = torch.matmul(query_detached, all_negs.T) / entropy_temp
        probs = F.softmax(sim, dim=-1)

        # Generated negatives' probabilities
        gen_probs = probs[num_mined:]

        # Component 1: Entropy maximization within generated negatives
        gen_probs_normalized = gen_probs / (gen_probs.sum() + 1e-10)
        log_gen_probs = torch.log(gen_probs_normalized + 1e-10)
        gen_entropy = -torch.sum(gen_probs_normalized * log_gen_probs)

        # Component 2: Probability balance constraint
        expected_gen_prob = num_gen_actual / (num_mined + num_gen_actual)
        actual_gen_prob = gen_probs.sum()
        prob_balance = (actual_gen_prob - expected_gen_prob) ** 2

        # Combined per-sample entropy loss
        sample_entropy_loss = -gen_entropy + prob_balance

        entropy_losses.append(sample_entropy_loss)
        gen_entropies.append(gen_entropy.item())
        actual_gen_probs.append(actual_gen_prob.item())
        gen_counts.append(num_gen_actual)
        all_gen_embs.append(gen)
        all_mined_embs.append(mined)
        all_query_embs.append(query)

    # Average entropy loss
    if len(entropy_losses) > 0:
        entropy_loss = torch.stack(entropy_losses).mean()
    else:
        entropy_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # ========== Logging ==========
    if not hasattr(infonce_entropy_loss, '_step_counter'):
        infonce_entropy_loss._step_counter = 0
    infonce_entropy_loss._step_counter += 1

    if infonce_entropy_loss._step_counter % log_interval == 0:
        with torch.no_grad():
            B_valid = len(all_gen_embs)
            if B_valid > 0:
                mean_gen_entropy = np.mean(gen_entropies)
                mean_actual_gen_prob = np.mean(actual_gen_probs)
                mean_num_gen = np.mean(gen_counts)
                max_gen_entropy = np.log(mean_num_gen) if mean_num_gen > 1 else 1.0

                all_gen_flat = torch.cat(all_gen_embs, dim=0)
                all_mined_flat = torch.cat(all_mined_embs, dim=0)
                gen_mined_sim = (all_gen_flat @ all_mined_flat.T).mean().item()

                gen_entropy_ratio = mean_gen_entropy / max_gen_entropy if max_gen_entropy > 0 else 0
                expected_gen_prob = mean_num_gen / (num_mined + mean_num_gen) if mean_num_gen > 0 else 0

                total = main_loss.item() + entropy_weight * entropy_loss.item()
                logger.info(
                    f"[Entropy] step={infonce_entropy_loss._step_counter} | "
                    f"main={main_loss.item():.4f}, ent_loss={entropy_loss.item():.4f}, total={total:.4f} | "
                    f"valid={B_valid}/{B}, avg_gen={mean_num_gen:.1f} | "
                    f"gen_entropy={mean_gen_entropy:.3f}/{max_gen_entropy:.3f} (ratio={gen_entropy_ratio:.3f}) | "
                    f"neg_prob: gen={mean_actual_gen_prob:.4f} (expect={expected_gen_prob:.4f}) | "
                    f"emb_sim: g-m={gen_mined_sim:.3f}"
                )

    total_loss = main_loss + entropy_weight * entropy_loss
    return total_loss


# =============================================================================
# Registration for MS-Swift
# Add to loss_mapping in swift/plugin/loss.py:
#   'infonce_entropy': infonce_entropy_loss
# =============================================================================
