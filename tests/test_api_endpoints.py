from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "ok" in data
    assert "model_loaded" in data


def test_predict_fallback_ok():
    r = client.post("/predict", json={"symbol": "AAPL", "threshold": 0.5})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"direction_probability", "direction_decision", "level_mean", "level_variance"}
    assert 0.0 <= data["direction_probability"] <= 1.0

