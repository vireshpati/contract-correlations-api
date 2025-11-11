"""Database schema."""
from sqlalchemy import Column, String, Float, Text, Boolean, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ContractCorrelation(Base):
    __tablename__ = "contract_correlations"

    id = Column(String, primary_key=True)
    contract_a_title = Column(String)
    contract_a_venue = Column(String)
    contract_b_title = Column(String)
    contract_b_venue = Column(String)
    underlying_event_correlation = Column(Float)
    correlation_type = Column(String)
    correlation_strength = Column(String)
    correlation_reasoning = Column(Text)
    analysis_confidence = Column(Float)
    is_active = Column(Boolean)
