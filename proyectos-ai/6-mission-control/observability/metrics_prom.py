"""
Métricas Prometheus para Mission Control.

Expone métricas de latencia por endpoint, tokens consumidos,
costo estimado y drift de calidad del sistema AI completo.
"""

from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Métricas de latencia por endpoint
# ---------------------------------------------------------------------------
REQUEST_LATENCY = Histogram(
    "mission_control_request_latency_seconds",
    "Latencia de requests por endpoint",
    labelnames=["endpoint", "method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

REQUEST_COUNT = Counter(
    "mission_control_requests_total",
    "Total de requests procesados",
    labelnames=["endpoint", "method", "status"],
)

# ---------------------------------------------------------------------------
# Métricas de tokens consumidos
# ---------------------------------------------------------------------------
TOKENS_INPUT = Counter(
    "mission_control_tokens_input_total",
    "Total de tokens de entrada consumidos",
    labelnames=["model", "endpoint"],
)

TOKENS_OUTPUT = Counter(
    "mission_control_tokens_output_total",
    "Total de tokens de salida generados",
    labelnames=["model", "endpoint"],
)

# ---------------------------------------------------------------------------
# Métricas de costo estimado (USD)
# ---------------------------------------------------------------------------
COST_ESTIMATED = Counter(
    "mission_control_cost_usd_total",
    "Costo acumulado estimado en USD",
    labelnames=["model"],
)

COST_GAUGE = Gauge(
    "mission_control_cost_usd_current_session",
    "Costo acumulado de la sesión actual en USD",
)

# ---------------------------------------------------------------------------
# Métricas de calidad y drift
# ---------------------------------------------------------------------------
QUALITY_SCORE = Gauge(
    "mission_control_quality_score",
    "Score de calidad actual del sistema (0-1)",
    labelnames=["component"],
)

DRIFT_SCORE = Gauge(
    "mission_control_drift_score",
    "Score de drift de calidad vs baseline (0 = sin drift)",
    labelnames=["component"],
)

QUALITY_EVALUATIONS = Counter(
    "mission_control_quality_evaluations_total",
    "Total de evaluaciones de calidad realizadas",
    labelnames=["component", "result"],
)

# ---------------------------------------------------------------------------
# Métricas de guardrails
# ---------------------------------------------------------------------------
GUARDRAIL_BLOCKED = Counter(
    "mission_control_guardrail_blocked_total",
    "Requests bloqueados por guardrails",
    labelnames=["stage", "reason"],
)

# ---------------------------------------------------------------------------
# Precio por modelo (USD por 1M tokens) — referencia 2026
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
    "mistral:7b": {"input": 0.0, "output": 0.0},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcula el costo en USD para un par de llamadas dado el modelo."""
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return round(cost, 8)


def record_tokens(model: str, endpoint: str, input_tokens: int, output_tokens: int) -> float:
    """Registra tokens y retorna el costo calculado."""
    TOKENS_INPUT.labels(model=model, endpoint=endpoint).inc(input_tokens)
    TOKENS_OUTPUT.labels(model=model, endpoint=endpoint).inc(output_tokens)

    cost = compute_cost(model, input_tokens, output_tokens)
    COST_ESTIMATED.labels(model=model).inc(cost)
    COST_GAUGE.inc(cost)
    return cost


def record_quality(component: str, score: float) -> None:
    """Registra un score de calidad (0-1) para un componente."""
    clamped = max(0.0, min(1.0, score))
    QUALITY_SCORE.labels(component=component).set(clamped)


def record_drift(component: str, baseline: float, current: float) -> None:
    """Registra el drift de calidad vs un baseline."""
    drift = abs(baseline - current) / baseline if baseline else 0.0
    DRIFT_SCORE.labels(component=component).set(round(drift, 4))


def record_guardrail_block(stage: str, reason: str) -> None:
    """Registra un evento de bloqueo por guardrails."""
    GUARDRAIL_BLOCKED.labels(stage=stage, reason=reason).inc()


def track_latency(endpoint: str, method: str = "POST") -> Callable[..., Any]:
    """Decorador que mide latencia de una función y la expone a Prometheus."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = "200"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "500"
                raise
            finally:
                elapsed = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)
                REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()

        return wrapper

    return decorator


def start_metrics_server(port: int | None = None) -> None:
    """Inicia el servidor HTTP de Prometheus para scraping."""
    resolved_port = port or int(os.getenv("METRICS_PORT", "9091"))
    start_http_server(resolved_port)
    print(f"[metrics] Prometheus metrics server listening on :{resolved_port}/metrics")


if __name__ == "__main__":
    start_metrics_server()
    print("[metrics] Expose metrics at http://localhost:9091/metrics")
    import time as _time

    while True:
        record_tokens("gpt-4o", "/api/chat", 120, 350)
        record_quality("rag", 0.87)
        record_drift("rag", 0.90, 0.87)
        _time.sleep(5)
