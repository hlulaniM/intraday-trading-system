from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "ok" in data


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "metrics_state" in data


