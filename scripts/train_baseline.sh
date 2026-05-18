#!/bin/bash
# =============================================================================
# Baseline Training Script: Standard InfoNCE
# =============================================================================
# Trains embedding models with standard InfoNCE loss (no entropy auxiliary loss).
# Sequentially trains on: hotpotqa, mmarco, trivia, nq
#
# Usage:
#   bash scripts/train_baseline.sh
#   # Or run in background:
#   nohup bash scripts/train_baseline.sh > logs/train_baseline.log 2>&1 &
# =============================================================================

# Environment
# Optional: set HF_ENDPOINT before running if you need a HuggingFace mirror.
export MASTER_PORT=${MASTER_PORT:-29512}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Model
MODEL_NAME="Qwen/Qwen3-0.6B"

# =========================================
# InfoNCE Loss Parameters
# =========================================
export INFONCE_USE_BATCH=false
export INFONCE_TEMPERATURE=0.05
export INFONCE_MASK_FAKE_NEGATIVE=true

# =========================================
# Entropy Loss Parameters: DISABLED
# =========================================
export ENTROPY_TEMPERATURE=0
export ENTROPY_WEIGHT=0.0
export NUM_MINED_NEGATIVES=0
export NUM_GENERATED_NEGATIVES=0
export CONFUSION_LOG_INTERVAL=10

# =========================================
# Data Configuration
# =========================================
# IMPORTANT: Update this path to your data directory
DATA_BASE_DIR="./data/train_data"

# Negative types to evaluate
# Possible values: mine_15, random_15, genneg_15, mine_15_genneg3_syneg
DATA_TYPE_LIST=("mine_15" "random_15" "genneg_15")

# Datasets
DATASET_LIST=("hotpotqa" "mmarco" "trivia" "nq")

# Dataset -> Evaluation Task mapping
declare -A EVAL_TASK_MAP
EVAL_TASK_MAP["hotpotqa"]="HotpotQAHardNegatives"
EVAL_TASK_MAP["mmarco"]="MMarcoRetrieval"
EVAL_TASK_MAP["trivia"]="TriviaQAHardNegatives"
EVAL_TASK_MAP["nq"]="NQRetrieval"

# =========================================
# Training Configuration
# =========================================
BATCH_SIZE=4
NUM_GPUS=8
OUTPUT_BASE="./ckpt"
EVAL_OUTPUT_BASE="./results"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

# Status tracking
BATCH_STATUS_FILE="${LOG_DIR}/batch_status_$(date +%Y%m%d_%H%M%S).log"
echo "=============================================="  | tee "$BATCH_STATUS_FILE"
echo "Batch training started: $(date)"  | tee -a "$BATCH_STATUS_FILE"
echo "Data types: ${DATA_TYPE_LIST[*]}"  | tee -a "$BATCH_STATUS_FILE"
echo "Datasets: ${DATASET_LIST[*]}"  | tee -a "$BATCH_STATUS_FILE"
echo "=============================================="  | tee -a "$BATCH_STATUS_FILE"

# =========================================
# Training + Evaluation Function
# =========================================
train_and_eval() {
    local DATASET=$1
    local EVAL_TASK=$2
    local DATA_TYPE=$3

    local ds_file="${DATA_BASE_DIR}/${DATASET}_${DATA_TYPE}.jsonl"
    if [ ! -f "$ds_file" ]; then
        echo "Error: Data file not found: $ds_file"
        return 1
    fi

    local TRAIN_OUTPUT_DIR="${OUTPUT_BASE}/baseline_${DATASET}_${DATA_TYPE}"

    echo "=============================================="
    echo "Training: $DATASET ($DATA_TYPE)"
    echo "Data: $ds_file"
    echo "Output: $TRAIN_OUTPUT_DIR"
    echo "=============================================="

    # ---- Training ----
    # Using ms-swift framework. Replace with your training framework.
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=$NUM_GPUS \
    swift sft \
        --model $MODEL_NAME \
        --task_type embedding \
        --model_type qwen3 \
        --template qwen3_emb \
        --train_type full \
        --dataset "$ds_file" \
        --eval_strategy steps \
        --split_dataset_ratio 0.05 \
        --output_dir "$TRAIN_OUTPUT_DIR" \
        --eval_steps 10 \
        --num_train_epochs 10 \
        --early_stop_interval 3 \
        --save_steps 10 \
        --per_device_train_batch_size $BATCH_SIZE \
        --per_device_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-5 \
        --max_model_len 8192 \
        --max_length 8192 \
        --warmup_ratio 0.1 \
        --loss_type infonce \
        --label_names labels \
        --use_chat_template false \
        --dataloader_drop_last true 2>&1 | tee /tmp/train_${DATASET}_${DATA_TYPE}.log

    local train_status=${PIPESTATUS[0]}

    # ---- Find best checkpoint ----
    local BEST_CKPT=$(grep -oP 'best_model_checkpoint: \K[^\s]+' /tmp/train_${DATASET}_${DATA_TYPE}.log | tail -1)

    if [ -z "$BEST_CKPT" ]; then
        echo "Warning: No best checkpoint found"
        local LATEST_VERSION=$(ls -td ${TRAIN_OUTPUT_DIR}/v*-* 2>/dev/null | head -1)
        if [ -n "$LATEST_VERSION" ]; then
            BEST_CKPT=$(ls -td ${LATEST_VERSION}/checkpoint-* 2>/dev/null | head -1)
        fi
    fi

    # ---- Evaluation ----
    if [ -n "$BEST_CKPT" ]; then
        local CKPT_STEP=$(basename "$BEST_CKPT" | sed 's/checkpoint-//')
        echo "Evaluating checkpoint: $BEST_CKPT"

        python -m src.evaluation.run_mteb \
            --model "$BEST_CKPT" \
            --model_name "baseline-${DATASET}-${DATA_TYPE}-${CKPT_STEP}" \
            --precision fp16 \
            --output_dir "${EVAL_OUTPUT_BASE}/baseline-${DATASET}-${DATA_TYPE}" \
            --batch_size 16 \
            --tasks "$EVAL_TASK"
    fi

    rm -f /tmp/train_${DATASET}_${DATA_TYPE}.log
    echo "[$DATA_TYPE][$DATASET] Status: Done (train=$train_status)" >> "$BATCH_STATUS_FILE"
}

# =========================================
# Main Loop
# =========================================
for data_type in "${DATA_TYPE_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
        eval_task="${EVAL_TASK_MAP[$dataset]}"
        echo ">>> Processing: $dataset ($data_type)"
        (train_and_eval "$dataset" "$eval_task" "$data_type")
    done
done

echo "=============================================="  | tee -a "$BATCH_STATUS_FILE"
echo "All training complete: $(date)"  | tee -a "$BATCH_STATUS_FILE"
echo "=============================================="  | tee -a "$BATCH_STATUS_FILE"
