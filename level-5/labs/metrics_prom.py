# level-5/labs/metrics_prom.py
"""
Observabilidad con formato Prometheus: metricas estandar para scraping.

Reemplaza al metrics.py de Level 2: en vez de un JSON propio, expone
metricas en el formato de texto que Prometheus entiende (# HELP / # TYPE),
usando prometheus_client (Counter, Histogram).

Usage:
    from metrics_prom import registrar_metricas
    registrar_metricas(app)   # middleware de FastAPI

    # en la ruta de /metrics:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
"""

import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total de peticiones HTTP recibidas",
    ["method", "path", "status"],
)
HTTP_LATENCIA = Histogram(
    "http_request_duration_seconds",
    "Duracion de las peticiones HTTP en segundos",
    ["path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)
LLM_RESPUESTAS = Counter(
    "librarian_answers_total",
    "Respuestas generadas por el RAG",
)


def registrar_metricas(app) -> None:
    """Agrega a la app el middleware que alimenta las metricas."""

    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        inicio = time.perf_counter()
        response = await call_next(request)
        duracion = time.perf_counter() - inicio

        path = request.url.path
        HTTP_REQUESTS.labels(
            method=request.method, path=path, status=response.status_code
        ).inc()
        HTTP_LATENCIA.labels(path=path).observe(duracion)
        return response


def endpoint_metrics() -> Response:
    """Handler listo para montar en GET /metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)