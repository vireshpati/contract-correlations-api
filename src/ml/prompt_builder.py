"""Pure functional prompt building for correlation prediction."""
from typing import Optional


def build_system_prompt() -> str:
    """Build the system prompt for the correlation prediction task."""
    return """You are an expert analyst specializing in prediction market contract correlations.

Your task is to analyze how two prediction market contracts correlate with each other.

Output your analysis in the following JSON format:
{
    "underlying_correlation": <float between -1 and 1>,
    "correlation_type": "<positive|negative|neutral>",
    "confidence": <float between 0 and 1>,
    "reasoning": "<detailed explanation>"
}

Correlation types:
- positive: Both contracts likely to resolve in the same direction (>0.3)
- negative: Contracts likely to resolve in opposite directions (<-0.3)
- neutral: Little to no correlation between contracts (-0.3 to 0.3)

Consider:
- Underlying events and causal relationships
- Common factors affecting both contracts
- Temporal alignment and timing
- Market dynamics and trader behavior"""


def build_user_prompt(
    contract_a: str,
    contract_b: str,
    rag_context: Optional[str] = None
) -> str:
    """
    Build the user prompt for correlation prediction.

    Args:
        contract_a: First contract description
        contract_b: Second contract description
        rag_context: Optional context from RAG retrieval

    Returns:
        Formatted user prompt string
    """
    prompt_parts = [
        "Analyze the correlation between these two prediction market contracts:\n",
        f"Contract A: {contract_a}",
        f"Contract B: {contract_b}\n",
    ]

    if rag_context:
        prompt_parts.extend([
            "Historical Context:",
            rag_context,
            "\nBased on the above context and your analysis, predict the correlation.\n"
        ])
    else:
        prompt_parts.append(
            "\nPredict the correlation between these contracts.\n"
        )

    prompt_parts.append(
        "Provide your analysis in the specified JSON format."
    )

    return "\n".join(prompt_parts)


def build_training_prompt(
    contract_a_title: str,
    contract_b_title: str,
    target_correlation: float,
    target_type: str,
    target_reasoning: str
) -> dict:
    """
    Build training prompt-completion pair for fine-tuning.

    Args:
        contract_a_title: Title of first contract
        contract_b_title: Title of second contract
        target_correlation: Ground truth correlation value
        target_type: Ground truth correlation type
        target_reasoning: Ground truth reasoning

    Returns:
        Dictionary with 'prompt' and 'completion' keys
    """
    system_msg = build_system_prompt()
    user_msg = build_user_prompt(contract_a_title, contract_b_title, rag_context=None)

    completion = {
        "underlying_correlation": target_correlation,
        "correlation_type": target_type,
        "confidence": 0.85,
        "reasoning": target_reasoning
    }

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": str(completion)}
        ]
    }
