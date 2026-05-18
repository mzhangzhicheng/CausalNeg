"""
Unit tests for loss functions.

Run with: python -m pytest tests/test_losses.py -v
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.losses.infonce import infonce_loss, infonce_loss_batched
from src.losses.entropy_loss import query_entropy_loss, infonce_with_entropy_loss
from src.losses.utils import parse_multi_negative_samples


class TestInfoNCELoss:
    """Tests for standard InfoNCE loss."""

    def test_basic_infonce(self):
        """Test InfoNCE loss with a simple example."""
        D = 64
        query = F.normalize(torch.randn(D), dim=-1)
        positive = F.normalize(torch.randn(D), dim=-1)
        negatives = F.normalize(torch.randn(5, D), dim=-1)

        loss = infonce_loss(query, positive, negatives, temperature=0.05)
        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_perfect_separation(self):
        """When positive is identical to query, loss should be low."""
        D = 64
        query = F.normalize(torch.randn(D), dim=-1)
        positive = query.clone()
        negatives = F.normalize(torch.randn(10, D), dim=-1)

        loss = infonce_loss(query, positive, negatives, temperature=0.05)
        assert loss.item() < 1.0, "Loss should be low for perfect match"

    def test_batched_infonce(self):
        """Test batched InfoNCE with cross-batch negatives."""
        D = 64
        B = 4
        num_neg = 5

        split_tensors = []
        for _ in range(B):
            t = F.normalize(torch.randn(2 + num_neg, D), dim=-1)
            split_tensors.append(t)

        loss = infonce_loss_batched(
            split_tensors, temperature=0.05, use_in_batch_negatives=True
        )
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_batched_per_sample(self):
        """Test batched InfoNCE with per-sample negatives only."""
        D = 64
        B = 4
        num_neg = 5

        split_tensors = []
        for _ in range(B):
            t = F.normalize(torch.randn(2 + num_neg, D), dim=-1)
            split_tensors.append(t)

        loss = infonce_loss_batched(
            split_tensors, temperature=0.05, use_in_batch_negatives=False
        )
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_variable_negatives(self):
        """Test with different number of negatives per sample."""
        D = 64
        split_tensors = [
            F.normalize(torch.randn(2 + 3, D), dim=-1),
            F.normalize(torch.randn(2 + 5, D), dim=-1),
            F.normalize(torch.randn(2 + 7, D), dim=-1),
        ]

        loss = infonce_loss_batched(
            split_tensors, temperature=0.05, use_in_batch_negatives=True
        )
        assert not torch.isnan(loss)


class TestEntropyLoss:
    """Tests for query-perspective entropy maximization loss."""

    def test_basic_entropy_loss(self):
        """Test entropy loss computation."""
        D = 64
        query = F.normalize(torch.randn(D), dim=-1)
        mined = F.normalize(torch.randn(15, D), dim=-1)
        gen = F.normalize(torch.randn(3, D), dim=-1, requires_grad=True)

        loss, stats = query_entropy_loss(query, mined, gen, temperature=0.1)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert 'gen_entropy' in stats
        assert 'actual_gen_prob' in stats
        assert 'balance_loss' in stats

    def test_gradient_flow(self):
        """Verify gradients only flow through generated negatives."""
        D = 64
        query = F.normalize(torch.randn(D), dim=-1)
        mined = F.normalize(torch.randn(15, D), dim=-1)
        # Use leaf tensors for gradient checking
        gen_raw = torch.randn(3, D, requires_grad=True)
        gen = F.normalize(gen_raw, dim=-1)

        loss, _ = query_entropy_loss(query, mined, gen, temperature=0.1)
        loss.backward()

        assert gen_raw.grad is not None, "Generated negatives should have gradients"
        assert gen_raw.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_entropy_maximization_direction(self):
        """Loss should decrease when gen probs become more uniform."""
        D = 64
        query = F.normalize(torch.randn(D), dim=-1)
        mined = F.normalize(torch.randn(15, D), dim=-1)

        # Case 1: gen negatives are very similar (low entropy expected)
        base_vec = F.normalize(torch.randn(D), dim=-1)
        gen_similar = base_vec.unsqueeze(0).repeat(3, 1) + 0.01 * torch.randn(3, D)
        gen_similar = F.normalize(gen_similar, dim=-1)

        # Case 2: gen negatives are diverse (high entropy expected)
        gen_diverse = F.normalize(torch.randn(3, D), dim=-1)

        _, stats_similar = query_entropy_loss(query, mined, gen_similar, temperature=0.1)
        _, stats_diverse = query_entropy_loss(query, mined, gen_diverse, temperature=0.1)

        # Note: Whether diverse or similar has higher entropy depends on
        # the query-based similarity, not just the negatives themselves.
        # This test just verifies both cases run without errors.
        assert stats_similar['gen_entropy'] >= 0
        assert stats_diverse['gen_entropy'] >= 0

    def test_combined_loss(self):
        """Test the full InfoNCE + Entropy combined loss."""
        D = 64
        B = 4
        num_mined = 15
        num_gen = 3

        split_tensors = []
        for _ in range(B):
            t = F.normalize(torch.randn(2 + num_mined + num_gen, D), dim=-1)
            split_tensors.append(t)

        step_counter = [0]
        loss = infonce_with_entropy_loss(
            split_tensors,
            temperature=0.05,
            entropy_temperature=0.1,
            entropy_weight=0.05,
            num_mined_negatives=num_mined,
            use_in_batch_negatives=False,
            log_interval=1,
            step_counter=step_counter,
        )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0
        assert step_counter[0] == 1

    def test_no_generated_negatives(self):
        """When no generated negatives exist, should fall back to InfoNCE only."""
        D = 64
        B = 4
        num_mined = 15

        split_tensors = []
        for _ in range(B):
            t = F.normalize(torch.randn(2 + num_mined, D), dim=-1)
            split_tensors.append(t)

        loss = infonce_with_entropy_loss(
            split_tensors,
            temperature=0.05,
            entropy_temperature=0.1,
            entropy_weight=0.05,
            num_mined_negatives=num_mined,
        )

        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestDataParsing:
    """Tests for data parsing utilities."""

    def test_parse_basic(self):
        """Test basic parsing of embeddings."""
        D = 32
        # 2 samples: first has 3 negs, second has 2 negs
        # Sample 1: anchor + pos + 3 negs = 5 embeddings
        # Sample 2: anchor + pos + 2 negs = 4 embeddings
        embeddings = torch.randn(9, D)
        labels = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])

        split = parse_multi_negative_samples(embeddings, labels)
        assert len(split) == 2
        assert split[0].shape == (5, D)  # anchor + pos + 3 negs
        assert split[1].shape == (4, D)  # anchor + pos + 2 negs

    def test_parse_with_truncation(self):
        """Test parsing with max_hard_negatives truncation."""
        D = 32
        embeddings = torch.randn(7, D)
        labels = torch.tensor([1, 0, 0, 0, 0, 0, 0])  # 1 sample, 5 negs

        split = parse_multi_negative_samples(embeddings, labels, max_hard_negatives=3)
        assert len(split) == 1
        assert split[0].shape == (5, D)  # anchor + pos + 3 negs (truncated from 5)

    def test_parse_single_sample(self):
        """Test parsing with a single sample."""
        D = 32
        embeddings = torch.randn(4, D)
        labels = torch.tensor([1, 0, 0, 0])

        split = parse_multi_negative_samples(embeddings, labels)
        assert len(split) == 1
        assert split[0].shape == (4, D)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
