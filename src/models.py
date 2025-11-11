"""Pydantic models for API request/response validation."""
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class PredictCorrelationRequest(BaseModel):
    """Request model for correlation prediction."""

    contract_a: str = Field(
        ...,
        min_length=5,
        description="First contract description or question",
        examples=["Will Bitcoin reach $100,000 by end of 2025?"]
    )
    contract_b: str = Field(
        ...,
        min_length=5,
        description="Second contract description or question",
        examples=["Will Ethereum reach $10,000 by end of 2025?"]
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG for context retrieval"
    )

    @field_validator('contract_a', 'contract_b')
    @classmethod
    def validate_contract_text(cls, v: str) -> str:
        """Ensure contract text is meaningful."""
        if len(v.strip()) < 5:
            raise ValueError("Contract description must be at least 5 characters")
        return v.strip()


class PredictCorrelationResponse(BaseModel):
    """Response model matching PRD specification."""

    underlying_correlation: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Correlation coefficient between -1 and 1"
    )
    correlation_type: str = Field(
        ...,
        description="Type of correlation: positive, negative, or neutral"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction (0-1)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why contracts correlate"
    )
    rag_context_used: bool = Field(
        default=False,
        description="Whether RAG context was used in prediction"
    )
    similar_examples_count: int = Field(
        default=0,
        description="Number of similar historical examples retrieved"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    database_connected: bool
    total_correlations: Optional[int] = None
