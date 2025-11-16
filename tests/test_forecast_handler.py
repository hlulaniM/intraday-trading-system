import types
from src.dashboard import app as dash_app_module


def test_forecast_handler_smoke(monkeypatch):
    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "direction_probability": 0.6,
                "direction_decision": 1,
                "level_mean": 123.45,
                "level_variance": 1.0,
            }

    monkeypatch.setattr(
        dash_app_module,
        "requests",
        types.SimpleNamespace(post=lambda *a, **k: Resp()),
    )

    out = dash_app_module.forecast_handler(1, 0, "AAPL", 0.5)
    assert len(out) == 7
    price_fig, dir_fig, details, status, targets, latency, asset = out
    assert hasattr(price_fig, "to_plotly_json")
    assert hasattr(dir_fig, "to_plotly_json")
    # Symbol should appear in details table
    assert "AAPL" in str(details)
    assert "Forecast OK" in status.children
    assert isinstance(latency, str) and latency.endswith(" ms")
    assert asset == "AAPL"

