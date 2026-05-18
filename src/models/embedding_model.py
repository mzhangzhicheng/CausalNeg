"""
Generic Embedding Model Wrapper.

This module provides a framework-agnostic embedding model implementation
that supports various transformer-based models (e.g., Qwen3, BERT, etc.)
with configurable pooling strategies and instruction templates.
"""

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel(nn.Module):
    """
    A general-purpose text embedding model built on top of HuggingFace transformers.
    
    Supports:
    - Multiple pooling strategies: last-token, first-token (CLS), mean pooling
    - Causal and bidirectional attention
    - L2 normalization
    - Instruction-based query encoding
    
    Args:
        model_name_or_path: HuggingFace model name or local path.
        pooler_type: Pooling strategy ('last', 'first', 'mean').
        do_norm: Whether to L2-normalize output embeddings.
        attn_type: Attention type ('causal' or 'bidirectional').
        max_length: Maximum sequence length.
        trust_remote_code: Whether to trust remote code for model loading.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        pooler_type: str = 'last',
        do_norm: bool = True,
        attn_type: str = 'causal',
        max_length: int = 8192,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        
        self.pooler_type = pooler_type
        self.do_norm = do_norm
        self.attn_type = attn_type
        self.max_length = max_length
    
    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        prompt: Optional[str] = None,
        device: str = 'cuda',
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of sentences into embeddings.
        
        Args:
            sentences: List of input texts.
            batch_size: Encoding batch size.
            prompt: Optional instruction prefix for queries.
            device: Target device.
            show_progress: Whether to show a progress bar.
        
        Returns:
            [N, D] tensor of embeddings.
        """
        self.eval()
        self.to(device)
        
        all_embeddings = []
        
        from tqdm import tqdm
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for start in iterator:
                batch = sentences[start:start + batch_size]
                if prompt:
                    batch = [prompt + text for text in batch]
                
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                ).to(device)
                
                embeddings = self.forward(**inputs)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        embeddings = self._pool(output.last_hidden_state, attention_mask)
        
        if self.do_norm:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def _pool(
        self, 
        hidden_state: torch.Tensor, 
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply pooling to get sentence embeddings."""
        if self.pooler_type == 'last':
            return self._pooling_last(hidden_state, attention_mask)
        elif self.pooler_type == 'first':
            return hidden_state[:, 0]
        elif self.pooler_type == 'mean':
            return self._pooling_mean(hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooler_type: {self.pooler_type}")
    
    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, 
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Last-token pooling (compatible with left-padding)."""
        mask = attention_mask
        left_padding = (mask[:, -1].sum() == mask.shape[0])
        if left_padding:
            return hidden_state[:, -1]
        else:
            sequence_lengths = mask.sum(dim=1) - 1
            batch_size = hidden_state.shape[0]
            return hidden_state[
                torch.arange(batch_size, device=hidden_state.device),
                sequence_lengths,
            ]
    
    @staticmethod
    def _pooling_mean(
        hidden_state: torch.Tensor, 
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over non-padded tokens."""
        attention_mask = attention_mask.float()
        lengths = attention_mask.sum(1)
        pooled = torch.einsum(
            'bsh,bs,b->bh',
            (hidden_state.float(), attention_mask, 1 / lengths),
        )
        return pooled
