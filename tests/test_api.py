"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app


@pytest.fixture
def client():
    """Test client for API."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


@patch("src.main.get_predictor")
@patch("src.main.get_db_session")
def test_health_endpoint(mock_db_session, mock_predictor, client):
    """Test health check endpoint."""
    # Mock predictor
    predictor = MagicMock()
    predictor.is_loaded = True
    mock_predictor.return_value = predictor

    # Mock database session
    session_mock = MagicMock()
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=None)
    mock_db_session.return_value = session_mock

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "database_connected" in data


@patch("src.main.get_predictor")
@patch("src.main.get_db_session")
def test_predict_correlation_endpoint(mock_db_session, mock_predictor, client, sample_contracts):
    """Test correlation prediction endpoint."""
    # Mock predictor
    predictor = MagicMock()
    predictor.predict.return_value = {
        "underlying_correlation": 0.75,
        "correlation_type": "positive",
        "confidence": 0.85,
        "reasoning": "Both contracts relate to cryptocurrency prices."
    }
    mock_predictor.return_value = predictor

    # Mock database session
    session_mock = MagicMock()
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=None)
    mock_db_session.return_value = session_mock

    response = client.post(
        "/predict-correlation",
        json={
            "contract_a": sample_contracts["contract_a"],
            "contract_b": sample_contracts["contract_b"],
            "use_rag": False
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "underlying_correlation" in data
    assert "correlation_type" in data
    assert "confidence" in data
    assert "reasoning" in data
    assert data["underlying_correlation"] == 0.75
    assert data["correlation_type"] == "positive"


def test_predict_correlation_validation(client):
    """Test request validation."""
    response = client.post(
        "/predict-correlation",
        json={
            "contract_a": "a",  # Too short
            "contract_b": "Will Ethereum reach $10k?"
        }
    )

    assert response.status_code == 422  # Validation error
