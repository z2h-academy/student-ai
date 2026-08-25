# level-2/labs/metrics.py
"""Observabilidad basica: logs estructurados y metricas por request.

Registra latencia y conteo de cada peticion, y expone un resumen en /metrics.
Las metricas viven en memoria (suficiente para un servicio chico).
"""

import logging
import time
from collections import defaultdict

logger = logging.getLogger("librarian")
logger.setLevel(logging.INFO)

_requests = 0
_total_latencia = 0.0
_errores = 0
_latencia_por_endpoint: dict[str, list[float]] = defaultdict(list)


def registrar_metricas(app) -> None:
    """Agrega un middleware a la app que mide latencia y cuenta peticiones."""

    @app.middleware("http")
    async def _middleware(request, call_next):
        global _requests, _total_latencia, _errores

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            _errores += 1
            logger.exception("error path=%s", request.url.path)
            raise
        finally:
            latencia = time.perf_counter() - start
            _requests += 1
            _total_latencia += latencia
            _latencia_por_endpoint[request.url.path].append(latencia)
            logger.info(
                "request path=%s status=%s latencia_ms=%.1f",
                request.url.path,
                getattr(response, "status_code", "?"),
                latencia * 1000,
            )

        return response


def metricas_resumen() -> dict:
    """Devuelve un resumen de las metricas acumuladas."""
    promedio = _total_latencia / _requests if _requests else 0.0
    por_endpoint = {
        path: {
            "count": len(values),
            "avg_ms": round((sum(values) / len(values)) * 1000, 1),
        }
        for path, values in _latencia_por_endpoint.items()
    }
    return {
        "requests": _requests,
        "errores": _errores,
        "latencia_promedio_ms": round(promedio * 1000, 1),
        "por_endpoint": por_endpoint,
    }
