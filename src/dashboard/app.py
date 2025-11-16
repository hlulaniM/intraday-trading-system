"""Dash dashboard for monitoring ingestion, metrics, and trades with richer visuals."""

from __future__ import annotations

import json
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import requests
import time
import logging
import traceback

MANIFEST_PATH = Path("data/raw/manifest.json")
BASELINE_METRICS = Path("data/processed/sequences/aapl_baselines_latest.json")
HYBRID_METRICS = Path("data/processed/sequences/aapl_hybrid_latest_metrics.json")
BACKTEST_RESULTS = Path("docs/backtests/aapl_backtest.json")
TRADES_LOG = Path("logs/trades.jsonl")
BEST_MODELS_DIR = Path("data/processed/models")
BACKTEST_FILES = {
    "AAPL": Path("docs/backtests/aapl_backtest.json"),
    "TSLA": Path("docs/backtests/tsla_backtest.json"),
    "BTC/USD": Path("docs/backtests/btc_backtest.json"),
    "ETH/USD": Path("docs/backtests/eth_backtest.json"),
}


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


def kpi_card(title: str, value: str, subtitle: str | None = None, color: str = "primary") -> html.Div:
    return html.Div(
        className=f"card text-bg-{color} mb-3",
        children=[
            html.Div(className="card-body", children=[html.H6(title, className="card-title mb-2"), html.H3(value, className="card-text mb-0"), html.Small(subtitle or "")]),
        ],
        style={"minWidth": "160px"},
    )


def _extract_direction_level(metrics: dict) -> tuple[dict, dict]:
    d = metrics.get("direction", {}) if isinstance(metrics, dict) else {}
    l = metrics.get("level", {}) if isinstance(metrics, dict) else {}
    return d, l


def metrics_cards():
    baseline = load_json(BASELINE_METRICS) or {}
    hybrid = load_json(HYBRID_METRICS) or {}
    d_b, l_b = _extract_direction_level(baseline)
    d_h, l_h = _extract_direction_level(hybrid)

    baseline_row = html.Div(
        className="d-flex flex-wrap gap-2",
        children=[
            kpi_card("Baseline • Dir Acc", f"{d_b.get('accuracy', 0):.2f}", "precision/recall/f1", "secondary"),
            kpi_card("Baseline • MAE", f"{l_b.get('mae', 0):.2f}", "level error", "secondary"),
            kpi_card("Baseline • RMSE", f"{l_b.get('rmse', 0):.2f}", "", "secondary"),
            kpi_card("Baseline • MAPE", f"{l_b.get('mape', 0):.3f}", "", "secondary"),
        ],
    )
    hybrid_row = html.Div(
        className="d-flex flex-wrap gap-2",
        children=[
            kpi_card("Hybrid • Dir Acc", f"{d_h.get('accuracy', 0):.2f}", "precision/recall/f1", "success"),
            kpi_card("Hybrid • MAE", f"{l_h.get('mae', 0):.2f}", "level error", "success"),
            kpi_card("Hybrid • RMSE", f"{l_h.get('rmse', 0):.2f}", "", "success"),
            kpi_card("Hybrid • MAPE", f"{l_h.get('mape', 0):.3f}", "", "success"),
        ],
    )
    return html.Div([baseline_row, html.Hr(), hybrid_row])


def backtest_chart():
    # Build a grouped bar across symbols comparing baseline vs hybrid PnL
    symbols = []
    baseline = []
    hybrid = []
    for sym, path in BACKTEST_FILES.items():
        if not path.exists():
            continue
        data = load_json(path) or {}
        b = (data.get("baseline_random_forest") or {}).get("total_pnl", None)
        h = (data.get("hybrid") or {}).get("total_pnl", None)
        if b is None or h is None:
            continue
        symbols.append(sym)
        baseline.append(b)
        hybrid.append(h)

    fig = go.Figure()
    if symbols:
        fig.add_bar(name="Baseline (RF)", x=symbols, y=baseline, marker_color="#8E8E93", hovertemplate="Baseline PnL: %{y:.2f}<extra>%{x}</extra>")
        fig.add_bar(name="Hybrid", x=symbols, y=hybrid, marker_color="#10A37F", hovertemplate="Hybrid PnL: %{y:.2f}<extra>%{x}</extra>")
    else:
        fig.add_annotation(text="No backtest files found. Run backtests or place JSONs in docs/backtests/", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(
        barmode="group",
        title="Backtest Total PnL by Symbol",
        yaxis_title="Total PnL",
        template="plotly_dark",
        margin=dict(l=30, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(tickformat=".2f")
    )
    return fig


def equity_curve(symbol: str) -> go.Figure:
    path = BACKTEST_FILES.get(symbol)
    data = load_json(path) if path else None
    curve = (data or {}).get("hybrid", {}).get("equity_curve", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=curve, mode="lines", name=f"{symbol} hybrid", line=dict(color="#10A37F")))
    fig.update_layout(
        title=f"Equity Curve • {symbol}",
        xaxis_title="Trade #",
        yaxis_title="Cumulative PnL",
        template="plotly_white",
        margin=dict(l=30, r=20, t=40, b=40),
    )
    return fig


def trades_table():
    trades = load_trades()
    if not trades:
        return html.Div(
            className="alert alert-warning",
            children=[
                html.Strong("No trades yet. "),
                html.Span("Send a TradingView webhook to "),
                html.Code("POST /webhook"),
                html.Span(" or trigger your strategy to see executions here."),
            ],
        )
    header = html.Thead(
        html.Tr(
            [html.Th(col) for col in ["Time", "Symbol", "Decision", "Confidence", "Order ID"]]
        )
    )
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(trade.get("timestamp")),
                    html.Td(trade.get("symbol")),
                    html.Td(trade.get("decision")),
                    html.Td(f"{trade.get('confidence', 0):.2f}"),
                    html.Td(trade.get("order_id") or "—"),
                ]
            )
            for trade in trades
        ]
    )
    return html.Div(className="table-responsive", children=[html.Table([header, body], className="table table-dark table-striped table-sm align-middle")])


def load_best_model_metrics():
    results = {}
    if not BEST_MODELS_DIR.exists():
        return results
    for sym in ["aapl", "tsla", "btc", "eth"]:
        file = BEST_MODELS_DIR / f"{sym}_hybrid_best_metrics.json"
        if file.exists():
            try:
                results[sym.upper()] = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return results


def best_models_table_and_chart():
    data = load_best_model_metrics()
    if not data:
        return html.Div("Best-model metrics not found."), dcc.Graph(figure=go.Figure())

    # Table
    header = html.Thead(
        html.Tr([html.Th(col) for col in ["Symbol", "Dir Acc", "Precision", "Recall", "F1", "Level MAE", "Level RMSE"]])
    )
    rows = []
    for sym, metrics in data.items():
        d = metrics.get("direction", {})
        l = metrics.get("level", {})
        rows.append(
            html.Tr(
                [
                    html.Td(sym),
                    html.Td(f"{d.get('accuracy', 0):.2f}"),
                    html.Td(f"{d.get('precision', 0):.2f}"),
                    html.Td(f"{d.get('recall', 0):.2f}"),
                    html.Td(f"{d.get('f1', 0):.2f}"),
                    html.Td(f"{l.get('mae', 0):.2f}"),
                    html.Td(f"{l.get('rmse', 0):.2f}"),
                ]
            )
        )
    table = html.Table([header, html.Tbody(rows)])

    # Bar chart for Level MAE
    symbols = list(data.keys())
    maes = [data[s]["level"]["mae"] for s in symbols]
    fig = px.bar(x=symbols, y=maes, labels={"x": "Symbol", "y": "Level MAE"}, title="Best-Model Level MAE by Symbol", color=symbols, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template="plotly_white", margin=dict(l=30, r=20, t=40, b=40), showlegend=False)
    return table, dcc.Graph(figure=fig)


# Use a Bootstrap stylesheet for nicer styling without extra dependency
external_stylesheets = ["https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/darkly/bootstrap.min.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
log = logging.getLogger("dashboard")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)
app.layout = html.Div(className="container py-4", children=[
    html.H2("Intraday Forecast Dashboard", className="mb-3"),
    dcc.Interval(id="refresh", interval=60_000, n_intervals=0),
    dcc.Tabs(id="tabs", value="forecast", children=[
        dcc.Tab(label="Forecast", value="forecast", children=[
            # Header/status bar
            html.Div(className="card bg-dark mb-3", children=[
                html.Div(className="card-body d-flex flex-wrap gap-3 align-items-center", children=[
                    html.Div([html.Small("Asset"), html.H5(id="hdr-asset", children="—")], className="me-4"),
                    html.Div([html.Small("Timeframe"), html.H5("1m")], className="me-4"),
                    html.Div([html.Small("Mode"), html.H5("LIVE")], className="me-4"),
                    html.Div([html.Small("Model"), html.H5("ACTIVE")], className="me-4"),
                    html.Div([html.Small("Latency"), html.H5(id="hdr-latency", children="— ms")]),
                ])
            ]),
            # Main area
            html.Div(className="row", children=[
                # Left input/prediction panel
                html.Div(className="col-lg-4", children=[
                    html.Div(className="card bg-dark mb-3", children=[
                        html.Div(className="card-body", children=[
                            html.Label("Enter Stock Ticker"),
                            dcc.Input(id="fc-symbol", type="text", value="AAPL", className="form-control"),
                            html.Label("Decision Threshold", className="mt-3"),
                            dcc.Slider(id="fc-threshold", min=0, max=1, step=0.01, value=0.5, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Button("Submit", id="fc-run", className="btn btn-primary mt-3"),
                            html.Div(id="fc-status", className="mt-2"),
                        ])
                    ]),
                    html.Div(className="card bg-dark mb-3", children=[
                        html.Div(className="card-header", children="Prediction Engine"),
                        html.Div(className="card-body", children=[
                            dcc.Graph(id="fc-direction", style={"height": "240px"}),
                            html.Div(id="fc-details"),
                            html.H6("Target Levels", className="mt-3"),
                            html.Div(id="fc-targets"),
                        ])
                    ]),
                ]),
                # Right main chart
                html.Div(className="col-lg-8", children=[
                    html.Div(className="card bg-dark mb-3", children=[
                        html.Div(className="card-header", children="Live Price & Prediction"),
                        html.Div(className="card-body", children=[
                            dcc.Graph(id="fc-price", style={"height": "420px"}),
                        ])
                    ]),
                    html.Div(className="row", children=[
                        html.Div(className="col-md-6", children=[
                            html.Div(className="card bg-dark mb-3", children=[
                        html.Div(className="card-header", children="Performance"),
                                html.Div(className="card-body", children=[html.Div(id="metrics-compact")]),
                            ])
                        ]),
                        html.Div(className="col-md-6", children=[
                            html.Div(className="card bg-dark mb-3", children=[
                                html.Div(className="card-header", children="Live Activity"),
                                html.Div(className="card-body", children=[html.Div(id="activity-panel")]),
                            ])
                        ]),
                    ]),
                ]),
            ]),
        ]),
        dcc.Tab(label="Overview", value="overview", children=[
            html.Div(className="row mt-3", children=[
                html.Div(className="col-md-6", children=[
                    html.H4("Model KPIs", className="mt-2"),
                    html.Div(id="metrics-panel"),
                    html.H4("Best Models", className="mt-4"),
                    html.Div(id="best-models-table", className="table-responsive"),
                    html.Div(id="best-models-chart", className="mt-2"),
                ]),
                html.Div(className="col-md-6", children=[
                    html.H4("Backtest Summary", className="mt-2"),
                    dcc.Graph(id="backtest-panel"),
                    html.H4("Recent Ingestion", className="mt-4"),
                    html.Div(id="manifest-panel", className="table-responsive"),
                ]),
            ]),
        ]),
        dcc.Tab(label="Backtests", value="backtests", children=[
            html.Div(className="row mt-3", children=[
                html.Div(className="col-md-4", children=[
                    dcc.Dropdown(
                        id="symbol-select",
                        options=[{"label": s, "value": s} for s in BACKTEST_FILES.keys()],
                        value="AAPL",
                        clearable=False,
                    ),
                ]),
                html.Div(className="col-md-8", children=[
                    dcc.Graph(id="equity-curve"),
                ]),
            ]),
        ]),
        dcc.Tab(label="Trades", value="trades", children=[
            html.Div(className="row mt-3", children=[
                html.Div(className="col-md-5", children=[
                    html.H5("Send Trade (Webhook)"),
                    html.Div(className="mb-2", children=[
                        html.Label("Symbol"),
                        dcc.Input(id="wh-symbol", type="text", placeholder="AAPL", value="AAPL", className="form-control"),
                    ]),
                    html.Div(className="mb-2", children=[
                        html.Label("Direction"),
                        dcc.Dropdown(
                            id="wh-direction",
                            options=[{"label": "Buy", "value": "buy"}, {"label": "Sell", "value": "sell"}],
                            value="buy",
                            clearable=False,
                        ),
                    ]),
                    html.Div(className="mb-2", children=[
                        html.Label("Confidence"),
                        dcc.Slider(id="wh-confidence", min=0, max=1, step=0.01, value=0.75, tooltip={"placement": "bottom", "always_visible": True}),
                    ]),
                    html.Button("Send", id="wh-send", className="btn btn-success"),
                    html.Div(id="wh-status", className="mt-2"),
                ]),
                html.Div(className="col-md-7", children=[
                    html.H5("Recent Trades"),
                    html.Div(id="trades-panel", className="mt-2"),
                ]),
            ]),
        ]),
    ]),
])


@app.callback(
    [
        Output("manifest-panel", "children"),
        Output("metrics-panel", "children"),
        Output("backtest-panel", "figure"),
        Output("activity-panel", "children"),
        Output("best-models-table", "children"),
        Output("best-models-chart", "children"),
    ],
    Input("refresh", "n_intervals"),
)
def refresh_layout(_: int):
    best_table, best_chart = best_models_table_and_chart()
    return manifest_table(), metrics_cards(), backtest_chart(), trades_table(), best_table, best_chart


@app.callback(
    Output("equity-curve", "figure"),
    Input("symbol-select", "value"),
)
def update_equity(symbol: str):
    try:
        return equity_curve(symbol or "AAPL")
    except Exception:
        fig = go.Figure(); fig.update_layout(template="plotly_dark")
        return fig

def _build_empty():
    fig = go.Figure(); fig.update_layout(template="plotly_dark")
    return fig


@app.callback(
    [
        Output("fc-price", "figure"),
        Output("fc-direction", "figure"),
        Output("fc-details", "children"),
        Output("fc-status", "children"),
        Output("fc-targets", "children"),
        Output("hdr-latency", "children"),
        Output("hdr-asset", "children"),
    ],
    # Only trigger on explicit submit to avoid firing before layout is ready
    Input("fc-run", "n_clicks"),
    [State("fc-symbol", "value"), State("fc-threshold", "value")],
)
def forecast_handler(n_clicks: int, symbol: str, threshold: float):
    start = time.perf_counter()
    sym = (symbol or "AAPL").upper()
    th = float(threshold or 0.5)
    try:
        log.info("forecast_handler called n_clicks=%s symbol=%s threshold=%s", n_clicks, sym, th)
        payload = {"symbol": sym, "threshold": th}
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=8)
        r.raise_for_status()
        data = r.json()
        log.info("predict ok: %s", data)
        level_mean = float(data.get("level_mean", 0))
        level_var = float(data.get("level_variance", 0))
        level_std = max(level_var, 0.0) ** 0.5
        # Simple predicted point and uncertainty whisker
        price_fig = go.Figure()
        upper = level_mean + 1.96 * level_std
        lower = level_mean - 1.96 * level_std
        # Use numeric x to avoid category/type issues
        price_fig.add_trace(go.Scatter(x=[0], y=[level_mean], mode="markers", marker=dict(color="#10A37F", size=10), name="Mean"))
        price_fig.add_trace(go.Scatter(x=[0, 0], y=[lower, upper], mode="lines", line=dict(color="orange"), name="95% CI"))
        # Support/resistance proxy bands
        price_fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=level_mean + level_std, y1=level_mean + level_std, line=dict(color="#2E86AB", dash="dot"))
        price_fig.add_shape(type="line", x0=-0.5, x1=1.5, y0=level_mean - level_std, y1=level_mean - level_std, line=dict(color="#B23A48", dash="dot"))
        price_fig.update_layout(template="plotly_dark", margin=dict(l=30, r=20, t=40, b=40), yaxis_title="Price", xaxis=dict(visible=False))
        p = float(data.get("direction_probability", 0))
        dir_fig = go.Figure()
        dir_fig.add_bar(x=["Down", "Up"], y=[1 - p, p], marker_color=["#EF553B", "#10A37F"])
        dir_fig.update_layout(template="plotly_dark", yaxis=dict(range=[0, 1]), margin=dict(l=30, r=20, t=40, b=40))
        details = html.Table(className="table table-dark table-sm", children=[
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody([
                html.Tr([html.Td("Symbol"), html.Td(sym)]),
                html.Tr([html.Td("Direction Prob"), html.Td(f"{p:.2%}")]),
                html.Tr([html.Td("Level Mean"), html.Td(f"{level_mean:.2f}")]),
                html.Tr([html.Td("Level Var"), html.Td(f"{level_var:.4f}")]),
            ])
        ])
        # Compute naive probabilities for reaching bands under normality
        import math
        def phi(z):  # survival function
            return 0.5 * (1 - math.erf(z / math.sqrt(2)))
        prob_res = phi((level_std) / level_std) if level_std > 0 else 0.5  # ~0.1587
        prob_sup = phi((level_std) / level_std) if level_std > 0 else 0.5
        targets_div = html.Div(children=[
            html.Div(className="progress mb-2", children=[
                html.Div(className="progress-bar bg-info", style={"width": f"{prob_res*100:.0f}%"},
                         children=f"Resistance ~ {prob_res*100:.0f}%")
            ]),
            html.Div(className="progress", children=[
                html.Div(className="progress-bar bg-danger", style={"width": f"{prob_sup*100:.0f}%"},
                         children=f"Support ~ {prob_sup*100:.0f}%")
            ]),
        ])
        latency_ms = (time.perf_counter() - start) * 1000
        status = html.Div("Forecast OK", className="alert alert-success py-1 my-2")
        return price_fig, dir_fig, details, status, targets_div, f"{latency_ms:.0f} ms", sym
    except Exception as e:
        log.error("forecast_handler error: %s\n%s", e, traceback.format_exc())
        empty = _build_empty()
        err = html.Div(f"Error: {e}", className="alert alert-danger py-1 my-2")
        return empty, empty, html.Div(), err, html.Div(), "— ms", sym

@app.callback(
    Output("wh-status", "children"),
    Input("wh-send", "n_clicks"),
    [State("wh-symbol", "value"), State("wh-direction", "value"), State("wh-confidence", "value")],
    prevent_initial_call=True,
)
def send_webhook(n_clicks: int, symbol: str, direction: str, confidence: float):
    try:
        payload = {"symbol": (symbol or "").upper(), "direction": direction, "confidence": float(confidence or 0.0), "source": "dashboard"}
        resp = requests.post("http://127.0.0.1:8000/webhook", json=payload, timeout=5)
        if resp.status_code >= 200 and resp.status_code < 300:
            status = html.Div("Webhook sent", className="alert alert-success py-1 my-2")
        else:
            status = html.Div(f"Webhook failed: {resp.status_code}", className="alert alert-danger py-1 my-2")
    except Exception as e:
        status = html.Div(f"Error: {e}", className="alert alert-danger py-1 my-2")
    return status

if __name__ == "__main__":
    # Dash >=2.16 deprecates run_server in favor of run
    app.run(host="0.0.0.0", port=8050, debug=False)
