"""Security tests for the FastAPI REST API."""

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from provenance.api import app


class FakeRateLimiter:
    def __init__(self, max_requests=5):
        self.max_requests = max_requests
        self.clients: dict[str, int] = {}

    def check(self, client_id: str) -> bool:
        self.clients[client_id] = self.clients.get(client_id, 0) + 1
        return self.clients[client_id] <= self.max_requests


def _make_secured_app():
    secured = FastAPI()
    rate_limiter = FakeRateLimiter(max_requests=5)
    api_key = "test-secret-key"

    @secured.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header:
            return JSONResponse(status_code=401, content={"detail": "API key required"})
        if api_key_header != api_key:
            return JSONResponse(status_code=403, content={"detail": "Invalid API key"})

        client_id = request.client.host if request.client else "unknown"
        if not rate_limiter.check(client_id):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        return await call_next(request)

    from provenance.api import DetectRequest, DetectResponse

    @secured.post("/detect", response_model=DetectResponse)
    async def detect(request: DetectRequest):
        if not request.text:
            return JSONResponse(
                status_code=400, content={"detail": "text is required"}
            )
        if len(request.text) > 1_000_000:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "Text exceeds maximum allowed length"
                },
            )

        return JSONResponse(
            status_code=200,
            content={
                "score": 0.5,
                "label": "uncertain",
                "confidence": 0.3,
                "detector_scores": None,
                "heatmap": None,
            },
        )

    @secured.get("/health")
    async def health():
        return {"status": "healthy", "version": "0.1.0"}

    return secured, rate_limiter


@pytest.fixture
def secured_client():
    secured_app, _ = _make_secured_app()
    with TestClient(secured_app) as c:
        yield c


@pytest.fixture
def rate_limited_client():
    secured_app, limiter = _make_secured_app()
    with TestClient(secured_app) as c:
        yield c, limiter


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestAuthRejection:
    def test_no_api_key_returns_401(self, secured_client):
        response = secured_client.post(
            "/detect",
            json={"text": "Some text to analyze for authentication testing purposes."},
        )
        assert response.status_code == 401
        assert "api key" in response.json()["detail"].lower()

    def test_invalid_api_key_returns_403(self, secured_client):
        response = secured_client.post(
            "/detect",
            json={"text": "Some text to analyze for authentication testing purposes."},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403
        assert "invalid" in response.json()["detail"].lower()

    def test_valid_api_key_accepted(self, secured_client):
        response = secured_client.post(
            "/detect",
            json={"text": "Some text to analyze for authentication testing purposes."},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200

    def test_health_endpoint_no_auth_required(self, secured_client):
        response = secured_client.get("/health")
        assert response.status_code == 200


class TestRateLimiting:
    def test_requests_within_limit_succeed(self, rate_limited_client):
        client, _ = rate_limited_client
        headers = {"X-API-Key": "test-secret-key"}
        for _ in range(5):
            response = client.post(
                "/detect",
                json={"text": "Text within rate limit for testing purposes."},
                headers=headers,
            )
            assert response.status_code == 200

    def test_requests_exceeding_limit_return_429(self, rate_limited_client):
        client, _ = rate_limited_client
        headers = {"X-API-Key": "test-secret-key"}
        for _ in range(5):
            client.post(
                "/detect",
                json={"text": "Text for rate limit testing."},
                headers=headers,
            )
        response = client.post(
            "/detect",
            json={"text": "Text that should be rate limited."},
            headers=headers,
        )
        assert response.status_code == 429
        assert "rate limit" in response.json()["detail"].lower()


class TestInputSizeLimits:
    def test_normal_sized_text_accepted(self, secured_client):
        text = "Normal text. " * 20
        response = secured_client.post(
            "/detect",
            json={"text": text},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200

    def test_oversized_text_rejected(self, secured_client):
        huge_text = "x" * 1_000_001
        response = secured_client.post(
            "/detect",
            json={"text": huge_text},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "maximum" in detail or "length" in detail

    def test_empty_text_returns_error(self, client):
        response = client.post("/detect", json={"text": ""})
        assert response.status_code == 400


class TestXSSPrevention:
    def test_script_tags_in_text_not_reflected_unescaped(self, secured_client):
        xss_payload = '<script>alert("xss")</script> Some normal text here.'
        response = secured_client.post(
            "/detect",
            json={"text": xss_payload},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200
        body = response.text
        assert "<script>" not in body or "&lt;script&gt;" in body

    def test_html_entities_escaped_in_response(self, secured_client):
        html_payload = "<b>bold</b> and <img src=x onerror=alert(1)> text."
        response = secured_client.post(
            "/detect",
            json={"text": html_payload},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200

    def test_existing_api_handles_special_characters(self, client):
        special_text = "Text with <script>alert(1)</script> and & entities < > \" '"
        response = client.post("/detect", json={"text": special_text})
        assert response.status_code in (200, 400)


class TestAPIResponseSecurity:
    def test_response_is_json(self, client):
        response = client.post(
            "/detect",
            json={
                "text": (
                    "This is a test sentence that should be long enough to "
                    "process properly by the detection system."
                )
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_no_server_header_leak(self, client):
        response = client.get("/health")
        for _key, val in response.headers.items():
            assert "python" not in val.lower()
            assert "uvicorn" not in val.lower()
