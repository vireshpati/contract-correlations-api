# Hybrid Loss Training: Generation + Regression

## Overview

We've implemented a hybrid training approach that combines:
1. **Generation Loss** (Cross-Entropy): Learn to generate valid JSON with reasoning
2. **Regression Loss** (MSE): Learn accurate correlation numerical values

This addresses the original problem where the model learned to generate text but not accurate numbers.

## Architecture

```
Input: Contract A + Contract B
    ↓
Llama 3.1 8B (QLoRA fine-tuning)
    ↓
Generate JSON: {
    "underlying_correlation": 0.75,
    "correlation_type": "positive",
    "confidence": 0.85,
    "reasoning": "..."
}
    ↓
Parse correlation value: 0.75
    ↓
Dual Loss:
    Loss = generation_loss + 10.0 * MSE(pred_correlation, true_correlation)
```

## New Components

### 1. `src/ml/training_utils.py`
Helper functions to:
- Extract correlation values from JSON text
- Parse predictions from model logits
- Compute regression loss (MSE/MAE/Huber)

### 2. `src/ml/semantic_rag.py`
FAISS-based semantic retrieval:
- Build vector index from Llama embeddings
- Fast nearest-neighbor search
- Retrieve similar contract pairs during training/inference

### 3. `RegressionAugmentedTrainer` (in `training.py`)
Custom Trainer class that:
- Computes standard generation loss
- Extracts predicted correlation from logits
- Adds weighted MSE loss on correlation values
- Backprops through both losses

### 4. `examples/build_faiss_index.py`
Script to:
- Load all 750 contract pairs
- Compute embeddings using Llama
- Build FAISS index for fast retrieval

## Usage

### Step 1: Build FAISS Index (One-time, ~10 min)

```bash
cd /path/to/contract-correlations-api

# Make sure dependencies are installed
pip install faiss-gpu

# Build index
python examples/build_faiss_index.py
```

This creates `./faiss_index/` with embeddings for all contract pairs.

### Step 2: Train with Hybrid Loss (~15-20 min)

```bash
# Train with default config (regression_loss_weight=10.0)
python examples/train_model.py

# Or with custom weight
# Edit examples/train_model.py to change:
# config = TrainingConfig(regression_loss_weight=20.0)
```

**What happens during training:**
- For each batch:
  1. Forward pass → generate JSON
  2. Compute generation loss (cross-entropy)
  3. Parse correlation from generated JSON
  4. Compute MSE between predicted and true correlation
  5. Combined loss = generation_loss + 10.0 * regression_loss
  6. Backprop through both

**Monitoring:**
```
{'loss': 2.1, 'generation_loss': 2.0, 'regression_loss': 0.01, 'total_loss': 2.1}
```

### Step 3: Evaluate

```bash
# Evaluate on test set
python examples/evaluate_model.py --model ./models/llama-3.1-8b-correlation-qlora --rag
```

## Expected Improvements

| Metric | Before (Generation Only) | After (Hybrid Loss) |
|--------|-------------------------|---------------------|
| **MAE** | 0.53 | 0.2-0.3 |
| **RMSE** | 0.68 | 0.25-0.4 |
| **R²** | 0.01 | 0.5-0.7 |
| **Accuracy ±0.2** | 21% | 60-75% |

## Configuration

Key parameters in `TrainingConfig`:

```python
regression_loss_weight: float = 10.0  # Weight for MSE loss
regression_loss_type: str = "mse"     # "mse", "mae", or "huber"
```

**Tuning the weight:**
- **Too low (e.g., 1.0)**: Reasoning stays aligned, but numbers less accurate
- **Too high (e.g., 100.0)**: Numbers very accurate, but reasoning might misalign
- **Sweet spot: 10.0-20.0**: Good balance

## Potential Issues & Solutions

### Issue: Regression loss computation fails
**Symptom:** Warnings "Regression loss computation failed"
**Cause:** JSON parsing fails (model outputs malformed JSON)
**Solution:** Model falls back to generation loss only. This should improve as training progresses.

### Issue: Numbers don't align with reasoning
**Symptom:** Reasoning says "weak" but number is 0.9
**Cause:** Regression loss pushes numbers to be accurate, text stays with original patterns
**Solution:** This is expected and acceptable. The number is the ground truth.

### Issue: FAISS index not found
**Symptom:** "Index file not found" error during training
**Solution:** Run `python examples/build_faiss_index.py` first

## Files Modified/Created

**Modified:**
- `src/ml/training.py` - Added `RegressionAugmentedTrainer` class
- `requirements.txt` - Added `faiss-gpu`

**Created:**
- `src/ml/training_utils.py` - Correlation extraction helpers
- `src/ml/semantic_rag.py` - FAISS-based RAG
- `examples/build_faiss_index.py` - Index builder script
- `HYBRID_LOSS_TRAINING.md` - This document

## Next Steps

1. **Build FAISS index** on H100
2. **Train model** with hybrid loss
3. **Evaluate** and compare to baseline
4. **Tune regression_loss_weight** if needed
5. **(Optional)** Integrate semantic RAG into training loop for context-aware training

## Theory: Why This Works

**Problem with generation-only:**
- Loss treats all token prediction errors equally
- Predicting "0.75" vs "0.35" has same loss as "0.75" vs "0.73"
- No incentive to be numerically accurate

**Solution with hybrid loss:**
- MSE directly penalizes numerical errors
- Gradient flows back through model to improve correlation predictions
- Generation loss maintains ability to explain reasoning
- Best of both worlds: accurate numbers + interpretable reasoning
