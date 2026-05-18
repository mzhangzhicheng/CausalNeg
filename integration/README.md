# Framework Integration Guide

This directory contains integration code for specific training frameworks.

## MS-Swift Integration

[MS-Swift](https://github.com/modelscope/ms-swift) is an efficient fine-tuning framework for large language models.

### Setup Steps

1. **Install MS-Swift**:
   ```bash
   pip install ms-swift
   ```

2. **Add the entropy loss plugin**:
   Copy `ms_swift_plugin.py` to the MS-Swift plugin directory, or add the `infonce_entropy_loss` function to `swift/plugin/loss.py`.

3. **Register the loss function**:
   In `swift/plugin/loss.py`, add to the `loss_mapping` dictionary:
   ```python
   loss_mapping = {
       # ... existing losses ...
       'infonce_entropy': infonce_entropy_loss,
   }
   ```

4. **Set environment variables** before training:
   ```bash
   export ENTROPY_TEMPERATURE=0.1
   export ENTROPY_WEIGHT=0.05
   export NUM_MINED_NEGATIVES=15
   export NUM_GENERATED_NEGATIVES=3
   ```

5. **Run training** with `--loss_type infonce_entropy`:
   ```bash
   swift sft \
       --model Qwen/Qwen3-0.6B \
       --task_type embedding \
       --loss_type infonce_entropy \
       ...
   ```

## Other Frameworks

The core loss function in `src/losses/entropy_loss.py` is framework-agnostic and can be integrated into any PyTorch-based training pipeline. The key function is `query_entropy_loss()`, which takes query, mined negative, and generated negative embeddings as input and returns the entropy maximization loss.

### Minimal Integration Example

```python
from src.losses import infonce_loss_batched, query_entropy_loss

# During training step:
main_loss = infonce_loss_batched(split_tensors, temperature=0.05)

entropy_loss = 0.0
count = 0
for tensor in split_tensors:
    query = tensor[0]
    mined = tensor[2:2+num_mined]
    gen = tensor[2+num_mined:]
    if gen.size(0) > 0:
        loss, _ = query_entropy_loss(query, mined, gen, temperature=0.1)
        entropy_loss += loss
        count += 1

if count > 0:
    entropy_loss /= count

total_loss = main_loss + 0.05 * entropy_loss
total_loss.backward()
```
