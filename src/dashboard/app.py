"""Dash dashboard for monitoring ingestion and model metrics."""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import dcc, html
import plotly.graph_objects as go

MANIFEST_PATH = Path("data/raw/manifest.json")
BASELINE_METRICS = Path("data/processed/sequences/aapl_baselines_latest.json")
HYBRID_METRICS = Path("data/processed/sequences/aapl_hybrid_latest_metrics.json")
BACKTEST_RESULTS = Path("docs/backtests/aapl_backtest.json")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def manifest_rows():
    data = load_json(MANIFEST_PATH) or []
    rows = data[-5:]
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in ["symbol", "start", "end", "rows"]])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(row.get("symbol")),
                            html.Td(row.get("start")),
                            html.Td(row.get("end")),
                            html.Td(row.get("rows")),
                        ]
                    )
                    for row in rows
                ]
            ),
        ]
    )


def metrics_cards():
    baseline = load_json(BASELINE_METRICS) or {}
    hybrid = load_json(HYBRID_METRICS) or {}
    cards = []
    if baseline:
        cards.append(html.Div([html.H4("Baseline"), html.Pre(json.dumps(baseline, indent=2))]))
    if hybrid:
        cards.append(html.Div([html.H4("Hybrid"), html.Pre(json.dumps(hybrid, indent=2))]))
    return html.Div(cards)


def backtest_chart():
    backtest = load_json(BACKTEST_RESULTS) or {}
    labels = list(backtest.keys())
    pnl = [backtest[k]["total_pnl"] for k in labels] if backtest else []
    fig = go.Figure(data=[go.Bar(x=labels, y=pnl)])
    fig.update_layout(title="Backtest PnL", yaxis_title="Total PnL")
    return dcc.Graph(figure=fig)


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H2("Intraday Forecast Dashboard"),
        html.H3("Recent Ingestion"),
        manifest_rows(),
        html.H3("Model Metrics"),
        metrics_cards(),
        html.H3("Backtest Summary"),
        backtest_chart(),
    ]
)


if __name__ == "__main__":
    app.run_server(debug=True)
