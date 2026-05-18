"""
MTEB Evaluation Script.

Evaluates embedding models on retrieval tasks from the MTEB benchmark.
Supports custom tasks (HotpotQA, NQ, TriviaQA, MMarco) and standard MTEB tasks.

Usage:
    python -m src.evaluation.run_mteb \
        --model /path/to/checkpoint \
        --tasks HotpotQA \
        --output_dir results/ \
        --batch_size 16 \
        --precision fp16
"""

import json
import sys
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import HfArgumentParser
import mteb

logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Evaluation arguments."""
    
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or HuggingFace model ID."},
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Model name for save path."},
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "Model kwargs as JSON string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "MTEB.run() kwargs as JSON string."},
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "Encode kwargs as JSON string."},
    )
    output_dir: Optional[str] = field(
        default='results',
        metadata={"help": "Output directory for results."},
    )
    tasks: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated task names."},
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "Encoding batch size."},
    )
    precision: str = field(
        default='fp16',
        metadata={"help": "Model precision: fp16, bf16, fp32."},
    )
    overwrite_results: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite existing results."},
    )
    
    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = self.tasks.split(',')
        for name in ('model', 'encode', 'run'):
            attr_name = name + '_kwargs'
            attr = getattr(self, attr_name)
            if attr is None:
                setattr(self, attr_name, dict())
            elif isinstance(attr, str):
                setattr(self, attr_name, json.loads(attr))


def get_model(model_path, model_name, precision='fp16', **kwargs):
    """
    Load embedding model for evaluation.
    
    This function supports loading models via the generic EmbeddingModel wrapper
    or any mteb-compatible model wrapper.
    """
    from src.models import EmbeddingModel
    
    # Determine torch dtype
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
    }
    torch_dtype = dtype_map.get(precision, torch.float16)
    
    model = EmbeddingModel(
        model_name_or_path=model_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    return model


def main():
    parser = HfArgumentParser(EvalArguments)
    args, *_ = parser.parse_args_into_dataclasses()
    logger.info(f"Evaluation args: {args}")
    
    # Load tasks
    tasks = mteb.get_tasks(tasks=args.tasks)
    logger.info(f"Selected {len(tasks)} tasks: {[t.metadata.name for t in tasks]}")
    
    # Load model
    model = get_model(
        args.model,
        args.model_name,
        precision=args.precision,
        **args.model_kwargs,
    )
    
    # Run evaluation
    encode_kwargs = args.encode_kwargs or {}
    encode_kwargs.update(batch_size=args.batch_size)
    
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model,
        output_folder=args.output_dir,
        encode_kwargs=encode_kwargs,
        overwrite_results=args.overwrite_results,
        **(args.run_kwargs or {}),
    )
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")
    return results


if __name__ == '__main__':
    main()
