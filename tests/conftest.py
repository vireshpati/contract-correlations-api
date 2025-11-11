"""Pytest fixtures for testing."""
import pytest
from unittest.mock import Mock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.schema import Base


@pytest.fixture
def mock_predictor():
    """Mock correlation predictor."""
    predictor = Mock()
    predictor.is_loaded = True
    predictor.predict.return_value = {
        "underlying_correlation": 0.75,
        "correlation_type": "positive",
        "confidence": 0.85,
        "reasoning": "Both contracts relate to cryptocurrency market movements."
    }
    return predictor


@pytest.fixture
def sample_contracts():
    """Sample contract pairs for testing."""
    return {
        "contract_a": "Will Bitcoin reach $100,000 by end of 2025?",
        "contract_b": "Will Ethereum reach $10,000 by end of 2025?"
    }


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def sample_correlation_data():
    """Sample correlation data from database."""
    return [
        {
            "contract_a_title": "Bitcoin to $100k",
            "contract_b_title": "Ethereum to $10k",
            "underlying_correlation": 0.8,
            "correlation_type": "positive",
            "correlation_strength": "strong",
            "reasoning": "Both are major cryptocurrencies affected by similar market forces.",
            "common_factors": "['crypto market', 'institutional adoption']",
            "causal_relationship": "indirect",
            "confidence": 0.85
        }
    ]
