# CausalNeg

Official code for **When Hard Negatives Hurt: Bridging the Generative-Discriminative Gap in Hard Negative Synthesis for Retrieval** (KDD 2026).

CausalNeg is a framework for training dense retrievers with synthesized hard negatives. It contains two complementary components:

- **CoT-guided counterfactual perturbation** for constructing hard negatives by decomposing query-document relevance into explicit information requirements and perturbing one requirement at a time.
- **Query-view entropy maximization** for suppressing source-dependent shortcuts when mined and generated negatives are mixed during contrastive training.

## Repository Structure

```text
.
├── configs/                    # Example training configurations
├── data_generation/            # CoT-guided counterfactual negative generation
├── examples/                   # Minimal JSONL data example
├── integration/                # MS-Swift integration plugin
├── scripts/                    # Training and evaluation entry points
├── src/
│   ├── evaluation/             # MTEB evaluation wrapper
│   ├── losses/                 # InfoNCE and query-view entropy losses
│   └── models/                 # Generic embedding model wrapper
├── tests/                      # Unit tests for loss functions
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/mzhangzhicheng/CausalNeg.git
cd CausalNeg
pip install -r requirements.txt
```

For the MS-Swift training scripts, also install MS-Swift following the official instructions:

```bash
pip install ms-swift
```

## Data Format

Training data follows the chat-style JSONL format used by MS-Swift embedding training. Each line contains one query, one positive document, and multiple negatives:

```json
{
  "messages": [
    {"role": "system", "content": "Given a question, retrieve Wikipedia passages that answer the question"},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "positive_messages": [
    [{"role": "user", "content": "Paris is the capital and most populous city of France."}]
  ],
  "negative_messages": [
    [{"role": "user", "content": "Mined negative 1"}],
    [{"role": "user", "content": "Mined negative 2"}],
    [{"role": "user", "content": "Generated negative 1"}]
  ]
}
```

When using query-view entropy maximization, negatives must be ordered as:

```text
[mined negatives..., generated negatives...]
```

Set `NUM_MINED_NEGATIVES` so the loss can split the two groups correctly. See `examples/example_data.jsonl` for a complete toy example.

## Hard Negative Generation

`data_generation/generate_hard_negatives.py` implements the paper's two-stage generation pipeline:

1. Decompose the query-positive pair into information requirements and disruption strategies.
2. Generate counterfactual hard negatives that violate selected requirements while matching the target corpus style.

The script supports OpenAI-compatible APIs, concurrent generation, retry/backoff, resume from an existing output file, failure logging, flexible JSONL input schemas, and dataset profiles for `hotpotqa`, `mmarco`, `nq`, `trivia`, and `generic`.

Example:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python data_generation/generate_hard_negatives.py \
  --dataset hotpotqa \
  --input data/raw/hotpotqa.jsonl \
  --output data/generated/hotpotqa_causalneg.jsonl \
  --model gpt-4.1 \
  --workers 4 \
  --max-items 100 \
  --max-negatives-per-query 3
```

Use `--dry-run` to inspect the prompts for the first valid sample without calling the API. The output JSONL keeps the original query/positive/candidates, the structured CausalNeg analysis, generated negatives, basic quality flags, and token-usage metadata.

## Training

The main method trains with InfoNCE plus the query-view entropy auxiliary loss:

```bash
bash scripts/train_entropy.sh
```

Key hyperparameters are configured through environment variables in the script:

```bash
export INFONCE_TEMPERATURE=0.05
export ENTROPY_TEMPERATURE=0.1
export ENTROPY_WEIGHT=0.05
export NUM_MINED_NEGATIVES=15
export NUM_GENERATED_NEGATIVES=3
```

The baseline script trains with standard InfoNCE:

```bash
bash scripts/train_baseline.sh
```

Both scripts assume data files under `./data/train_data/` and may need path, GPU, and model settings adjusted for your environment.

## MS-Swift Integration

Our experiments use the MS-Swift embedding training framework. To enable the entropy loss:

1. Copy or merge `integration/ms_swift_plugin.py` into the MS-Swift loss plugin file.
2. Register `infonce_entropy` in MS-Swift's loss mapping.
3. Run training with `--loss_type infonce_entropy`.

See `integration/README.md` for details.

The core loss in `src/losses/entropy_loss.py` is pure PyTorch and can also be integrated into other training loops.

## Evaluation

Evaluate checkpoints with MTEB-compatible retrieval tasks:

```bash
bash scripts/evaluate.sh /path/to/checkpoint HotpotQAHardNegatives
```

The paper evaluates on four retrieval benchmarks: HotpotQA, MS MARCO, TriviaQA, and Natural Questions.

