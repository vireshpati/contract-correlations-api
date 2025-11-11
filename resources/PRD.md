Task: Create a FastAPI application that predicts how prediction market contracts correlate with each other.

Data:  contract_correlations table (see notebooks/contract_correlations.ipynb)

Model: LLama 3.1 8B QLoRA fine-tuned on the contract_correlations table + RAG. From Hugging Face.

Expected Output:

{
    "underlying_correlation": 0.5,
    "correlation_type": "positive",
    "confidence": 0.95,
    "reasoning": "Both contracts are related to the same underlying event..."
}

Test: pytest




