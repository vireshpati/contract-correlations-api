"""Tests for Pydantic models."""
import pytest
from pydantic import ValidationError

from src.models import CorrelationRequest, CorrelationResponse


def test_correlation_request_valid():
    """Test valid correlation request."""
    request = CorrelationRequest(
        contract_a="Test contract A",
        contract_b="Test contract B",
        use_rag=True
    )
    assert request.contract_a == "Test contract A"
    assert request.contract_b == "Test contract B"
    assert request.use_rag is True


def test_correlation_request_default_rag():
    """Test correlation request with default RAG value."""
    request = CorrelationRequest(
        contract_a="Test contract A",
        contract_b="Test contract B"
    )
    assert request.use_rag is True  # Default value


def test_correlation_request_missing_fields():
    """Test correlation request with missing required fields."""
    with pytest.raises(ValidationError):
        CorrelationRequest(contract_a="Test")


def test_correlation_response_valid():
    """Test valid correlation response."""
    response = CorrelationResponse(
        underlying_correlation=0.75,
        correlation_type="positive",
        confidence=0.95,
        reasoning="Both contracts are related to crypto markets"
    )
    assert response.underlying_correlation == 0.75
    assert response.correlation_type == "positive"
    assert response.confidence == 0.95


def test_correlation_response_validation():
    """Test correlation response validation."""
    # Correlation out of range
    with pytest.raises(ValidationError):
        CorrelationResponse(
            underlying_correlation=1.5,  # Invalid
            correlation_type="positive",
            confidence=0.95,
            reasoning="Test"
        )

    # Confidence out of range
    with pytest.raises(ValidationError):
        CorrelationResponse(
            underlying_correlation=0.5,
            correlation_type="positive",
            confidence=1.5,  # Invalid
            reasoning="Test"
        )


def test_correlation_response_optional_rag_context():
    """Test correlation response with optional RAG context."""
    response = CorrelationResponse(
        underlying_correlation=0.5,
        correlation_type="neutral",
        confidence=0.8,
        reasoning="Test",
        rag_context_used=3
    )
    assert response.rag_context_used == 3
