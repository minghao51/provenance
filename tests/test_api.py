"""Tests for FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from provenance.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestDetectEndpoint:
    def test_detect_valid_text(self, client):
        response = client.post(
            "/detect",
            json={
                "text": "This is a test sentence that should be long enough to process properly by the detection system."
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "label" in data
        assert "confidence" in data

    def test_detect_empty_text(self, client):
        response = client.post("/detect", json={"text": ""})
        assert response.status_code == 400

    def test_detect_with_ensemble_strategy(self, client):
        response = client.post(
            "/detect",
            json={
                "text": "This is a test sentence that should be long enough to process properly by the detection system.",
                "ensemble_strategy": "uncertainty_aware",
            },
        )
        assert response.status_code == 200

    def test_detect_with_domain(self, client):
        response = client.post(
            "/detect",
            json={
                "text": "This is a test sentence that should be long enough to process properly by the detection system.",
                "domain": "prose",
            },
        )
        assert response.status_code == 200


class TestBatchEndpoint:
    def test_batch_detect(self, client):
        response = client.post(
            "/batch",
            json={
                "texts": [
                    "This is a test sentence that should be long enough to process properly.",
                    "Another test sentence that is also long enough for the detection system.",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_batch_detect_empty(self, client):
        response = client.post("/batch", json={"texts": []})
        assert response.status_code == 400


class TestDetectorsEndpoint:
    def test_list_detectors(self, client):
        response = client.get("/detectors")
        assert response.status_code == 200
        data = response.json()
        assert "detectors" in data


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
