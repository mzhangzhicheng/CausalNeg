#!/bin/bash
# =============================================================================
# Evaluation Script
# =============================================================================
# Evaluate trained checkpoints on retrieval benchmarks using MTEB.
#
# Usage:
#   bash scripts/evaluate.sh /path/to/checkpoint HotpotQAHardNegatives
#   bash scripts/evaluate.sh /path/to/checkpoint MMarcoRetrieval
# =============================================================================

CHECKPOINT_PATH=$1
EVAL_TASK=$2
OUTPUT_DIR=${3:-"./results"}
BATCH_SIZE=${4:-16}
PRECISION=${5:-"fp16"}

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$EVAL_TASK" ]; then
    echo "Usage: bash scripts/evaluate.sh <checkpoint_path> <eval_task> [output_dir] [batch_size] [precision]"
    echo ""
    echo "Available evaluation tasks:"
    echo "  - HotpotQAHardNegatives"
    echo "  - MMarcoRetrieval"
    echo "  - TriviaQAHardNegatives"
    echo "  - NQRetrieval"
    echo "  - Any MTEB retrieval task"
    exit 1
fi

CKPT_NAME=$(basename "$CHECKPOINT_PATH")
MODEL_NAME="eval-${CKPT_NAME}-${EVAL_TASK}"

echo "=============================================="
echo "Evaluation Configuration"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Task: $EVAL_TASK"
echo "  Output: ${OUTPUT_DIR}/${MODEL_NAME}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Precision: $PRECISION"
echo "=============================================="

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

python -m src.evaluation.run_mteb \
    --model "$CHECKPOINT_PATH" \
    --model_name "$MODEL_NAME" \
    --precision "$PRECISION" \
    --model_kwargs '{"max_length": 8192, "attn_type": "causal", "pooler_type": "last", "do_norm": true}' \
    --output_dir "${OUTPUT_DIR}/${MODEL_NAME}" \
    --batch_size "$BATCH_SIZE" \
    --tasks "$EVAL_TASK"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}/${MODEL_NAME}"
echo "=============================================="
