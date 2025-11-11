"""Tests for prompt builder module."""
import pytest
from src.ml.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    build_training_prompt
)


def test_build_system_prompt():
    """Test system prompt generation."""
    prompt = build_system_prompt()

    assert "correlation" in prompt.lower()
    assert "json" in prompt.lower()
    assert "positive" in prompt.lower()
    assert "negative" in prompt.lower()
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_build_user_prompt_without_rag():
    """Test user prompt without RAG context."""
    contract_a = "Will Bitcoin reach $100k?"
    contract_b = "Will Ethereum reach $10k?"

    prompt = build_user_prompt(contract_a, contract_b)

    assert contract_a in prompt
    assert contract_b in prompt
    assert "Contract A:" in prompt
    assert "Contract B:" in prompt
    assert isinstance(prompt, str)


def test_build_user_prompt_with_rag():
    """Test user prompt with RAG context."""
    contract_a = "Will Bitcoin reach $100k?"
    contract_b = "Will Ethereum reach $10k?"
    rag_context = "Similar analysis: BTC and ETH are correlated..."

    prompt = build_user_prompt(contract_a, contract_b, rag_context)

    assert contract_a in prompt
    assert contract_b in prompt
    assert rag_context in prompt
    assert "Historical Context:" in prompt


def test_build_training_prompt():
    """Test training prompt generation."""
    result = build_training_prompt(
        contract_a_title="Bitcoin to $100k",
        contract_b_title="Ethereum to $10k",
        target_correlation=0.75,
        target_type="positive",
        target_reasoning="Both are cryptocurrencies"
    )

    assert "messages" in result
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"
    assert "Bitcoin to $100k" in result["messages"][1]["content"]
