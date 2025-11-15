"""Dash dashboard for monitoring ingestion, metrics, and trades."""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

MANIFEST_PATH = Path("data/raw/manifest.json")
BASELINE_METRICS = Path("data/processed/sequences/aapl_baselines_latest.json")
HYBRID_METRICS = Path("data/processed/sequences/aapl_hybrid_latest_metrics.json")
BACKTEST_RESULTS = Path("docs/backtests/aapl_backtest.json")
TRADES_LOG = Path("logs/trades.jsonl")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_trades(limit: int = 10):
    if not TRADES_LOG.exists():
        return []
    lines = TRADES_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]


def manifest_table():
    data = load_json(MANIFEST_PATH) or []
    rows = data[-5:]
    header = html.Thead(html.Tr([html.Th(col) for col in ["symbol", "start", "end", "rows"]]))
    body = html.Tbody(
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
    )
    return html.Table([header, body])


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
    return fig


def trades_table():
    trades = load_trades()
    header = html.Thead(html.Tr([html.Th(col) for col in ["timestamp", "symbol", "decision", "confidence", "order_id"]]))
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(trade.get("timestamp")),
                    html.Td(trade.get("symbol")),
                    html.Td(trade.get("decision")),
                    html.Td(f"{trade.get('confidence', 0):.2f}"),
                    html.Td(trade.get("order_id")),
                ]
            )
            for trade in trades
        ]
    )
    return html.Table([header, body])


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H2("Intraday Forecast Dashboard"),
        dcc.Interval(id="refresh", interval=60_000, n_intervals=0),
        html.H3("Recent Ingestion"),
        html.Div(id="manifest-panel"),
        html.H3("Model Metrics"),
        html.Div(id="metrics-panel"),
        html.H3("Backtest Summary"),
        dcc.Graph(id="backtest-panel"),
        html.H3("Recent Auto-Trades"),
        html.Div(id="trades-panel"),
    ]
)


@app.callback(
    [
        Output("manifest-panel", "children"),
        Output("metrics-panel", "children"),
        Output("backtest-panel", "figure"),
        Output("trades-panel", "children"),
    ],
    Input("refresh", "n_intervals"),
)
def refresh_layout(_: int):
    return manifest_table(), metrics_cards(), backtest_chart(), trades_table()


if __name__ == "__main__":
    app.run_server(debug=True)
