import pytest
from fastapi.testclient import TestClient

from heart_mlops.api import app


@pytest.fixture
def client():
    # Entering this context runs FastAPI startup handlers.
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client):
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["positive", "negative"]
    assert 0 <= result["confidence"] <= 1
    assert set(result["probabilities"]) == {"0", "1"}
    assert abs(sum(result["probabilities"].values()) - 1) < 0.001


def test_invalid_age_is_rejected(client):
    payload = {
        "age": 200,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_features_are_rejected(client):
    response = client.post("/predict", json={"age": 63})
    assert response.status_code == 422

def test_metrics_endpoint(client):
    # Generate a request so the middleware records a metric.
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "heart_disease_http_requests_total" in response.text
    assert "heart_disease_http_request_duration_seconds" in response.text
