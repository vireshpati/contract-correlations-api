"""Tests for database query functions."""
import pytest
from src.db.queries import (
    extract_keywords,
    format_rag_context
)


def test_extract_keywords():
    """Test keyword extraction."""
    text = "Will Bitcoin reach $100,000 by the end of 2025?"

    keywords = extract_keywords(text)

    assert isinstance(keywords, list)
    assert "bitcoin" in keywords
    assert "reach" in keywords
    # Short words should be filtered
    assert "by" not in keywords
    assert "the" not in keywords


def test_extract_keywords_min_length():
    """Test keyword extraction with custom min length."""
    text = "Bitcoin ETH BTC"

    keywords = extract_keywords(text, min_length=3)

    assert "bitcoin" in keywords
    assert "btc" in keywords


def test_format_rag_context_empty():
    """Test RAG context formatting with empty list."""
    context = format_rag_context([])

    assert "No similar" in context
    assert isinstance(context, str)


def test_format_rag_context_with_data(sample_correlation_data):
    """Test RAG context formatting with data."""
    context = format_rag_context(sample_correlation_data)

    assert "Bitcoin to $100k" in context
    assert "Ethereum to $10k" in context
    assert "0.80" in context or "0.8" in context
    assert "positive" in context
    assert isinstance(context, str)
    assert len(context) > 50
