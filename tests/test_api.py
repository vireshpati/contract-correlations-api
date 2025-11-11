"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "rag_available" in data


def test_predict_correlation_validation(client):
    """Test correlation prediction with invalid input."""
    # Missing required fields
    response = client.post("/predict-correlation", json={})
    assert response.status_code == 422

    # Invalid field types
    response = client.post("/predict-correlation", json={
        "contract_a": 123,
        "contract_b": "Test"
    })
    assert response.status_code == 422


def test_predict_correlation_format(client):
    """Test correlation prediction response format."""
    response = client.post("/predict-correlation", json={
        "contract_a": "Will Bitcoin reach $100,000 by 2025?",
        "contract_b": "Will Ethereum reach $10,000 by 2025?",
        "use_rag": False
    })

    # May be 503 if model not loaded, or 200 if successful
    if response.status_code == 200:
        data = response.json()
        assert "underlying_correlation" in data
        assert "correlation_type" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert -1.0 <= data["underlying_correlation"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["correlation_type"] in ["positive", "negative", "neutral"]
    elif response.status_code == 503:
        # Model not loaded is acceptable in test environment
        assert "Model not loaded" in response.json()["detail"]
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")
