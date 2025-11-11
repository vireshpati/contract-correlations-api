"""API models."""
from pydantic import BaseModel, Field
from typing import Optional


class CorrelationRequest(BaseModel):
    """Request model for correlation prediction."""
    contract_a: str = Field(..., description="First contract description")
    contract_b: str = Field(..., description="Second contract description")
    use_rag: bool = Field(default=True, description="Whether to use RAG for context")


class CorrelationResponse(BaseModel):
    """Response model for correlation prediction."""
    underlying_correlation: float = Field(..., ge=-1.0, le=1.0, description="Correlation strength between -1 and 1")
    correlation_type: str = Field(..., description="Type: positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    reasoning: str = Field(..., description="Explanation of the correlation")
    rag_context_used: Optional[int] = Field(None, description="Number of RAG examples used")
