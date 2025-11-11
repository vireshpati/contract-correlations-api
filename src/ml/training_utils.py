"""Training utilities for hybrid loss computation."""
import json
import re
import torch
from typing import Optional


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text.

    Args:
        text: Text potentially containing JSON

    Returns:
        JSON string or None if not found
    """
    # Try to find JSON object with regex
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        return json_match.group()
    return None


def extract_correlation_from_text(text: str) -> Optional[float]:
    """
    Parse correlation value from JSON text.

    Args:
        text: Text containing JSON with underlying_correlation field

    Returns:
        Correlation value or None if parsing fails
    """
    try:
        json_str = extract_json_from_text(text)
        if json_str:
            data = json.loads(json_str)
            if "underlying_correlation" in data:
                return float(data["underlying_correlation"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return None


def extract_correlation_from_labels(labels: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Extract ground truth correlation from training labels.

    Args:
        labels: Token IDs of target sequence [batch_size, seq_len]
        tokenizer: Tokenizer to decode labels

    Returns:
        Tensor of correlation values [batch_size]
    """
    batch_size = labels.shape[0]
    correlations = []

    for i in range(batch_size):
        # Decode labels (ignore padding)
        label_ids = labels[i][labels[i] != -100]  # -100 is typically padding
        text = tokenizer.decode(label_ids, skip_special_tokens=True)

        # Extract correlation
        corr = extract_correlation_from_text(text)
        if corr is not None:
            correlations.append(corr)
        else:
            # Fallback to 0.0 if parsing fails
            correlations.append(0.0)

    return torch.tensor(correlations, dtype=torch.float32, device=labels.device)


def extract_correlation_from_logits(
    logits: torch.Tensor,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Extract predicted correlation from model logits.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        tokenizer: Tokenizer to decode predictions
        attention_mask: Optional attention mask

    Returns:
        Tensor of predicted correlation values [batch_size]
    """
    batch_size = logits.shape[0]
    correlations = []

    # Greedy decode: take argmax
    pred_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    for i in range(batch_size):
        # Decode prediction
        if attention_mask is not None:
            # Only decode non-padded tokens
            mask = attention_mask[i].bool()
            tokens = pred_tokens[i][mask]
        else:
            tokens = pred_tokens[i]

        text = tokenizer.decode(tokens, skip_special_tokens=True)

        # Extract correlation
        corr = extract_correlation_from_text(text)
        if corr is not None:
            correlations.append(corr)
        else:
            # Fallback to 0.0 if parsing fails
            correlations.append(0.0)

    return torch.tensor(correlations, dtype=torch.float32, device=logits.device)


def compute_regression_loss(
    pred_correlations: torch.Tensor,
    true_correlations: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Compute regression loss on correlation predictions.

    Args:
        pred_correlations: Predicted values [batch_size]
        true_correlations: Ground truth values [batch_size]
        loss_type: Type of loss ("mse", "mae", "huber")

    Returns:
        Loss tensor (scalar)
    """
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(pred_correlations, true_correlations)
    elif loss_type == "mae":
        return torch.nn.functional.l1_loss(pred_correlations, true_correlations)
    elif loss_type == "huber":
        return torch.nn.functional.huber_loss(pred_correlations, true_correlations)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
