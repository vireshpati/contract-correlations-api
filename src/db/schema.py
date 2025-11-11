"""Database schema definitions."""
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ContractCorrelation(Base):
    """Contract correlation table model."""

    __tablename__ = "contract_correlations"

    id = Column(String, primary_key=True)
    contract_a_id = Column(String)
    contract_a_title = Column(String)
    contract_a_venue = Column(String)
    contract_a_data = Column(Text)
    contract_b_id = Column(String)
    contract_b_title = Column(String)
    contract_b_venue = Column(String)
    contract_b_data = Column(Text)
    probability_correlation = Column(Float)
    underlying_event_correlation = Column(Float)
    correlation_type = Column(String)
    correlation_strength = Column(String)
    correlation_reasoning = Column(Text)
    price_correlation = Column(Float)
    volume_correlation = Column(Float)
    temporal_alignment = Column(String)
    expiry_correlation = Column(String)
    event_category_match = Column(Boolean)
    common_factors = Column(Text)
    causal_relationship = Column(String)
    analysis_confidence = Column(Float)
    anthropic_model = Column(String)
    analysis_prompt_version = Column(String)
    processing_time_ms = Column(Integer)
    is_active = Column(Boolean)
    needs_refresh = Column(Boolean)
    last_updated = Column(DateTime)
    created_at = Column(DateTime)
