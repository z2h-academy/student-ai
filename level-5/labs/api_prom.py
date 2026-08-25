# level-5/labs/api_prom.py
"""
La API librarian de Level 2 con metricas Prometheus estandar.

Misma funcionalidad que api_main.py (/health, /api/index, /api/ask)
pero el /metrics expone texto Prometheus (scrapeable) y suma el
contador de respuestas del RAG.

Usage:
    uvicorn api_prom:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from librarian_pkg.kb import indexar_base
from librarian_pkg.retrieval import recuperar
from librarian_pkg.generation import generar
from metrics_prom import LLM_RESPUESTAS, endpoint_metrics, registrar_metricas

app = FastAPI(title="librarian-api", version="prom")
registrar_metricas(app)


class PreguntaRequest(BaseModel):
    pregunta: str


class RespuestaResponse(BaseModel):
    pregunta: str
    respuesta: str
    fuentes: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Metricas en formato Prometheus (texto scrapeable)."""
    return endpoint_metrics()


@app.post("/api/index")
def api_index() -> dict:
    n = indexar_base()
    return {"indexados": n}


@app.post("/api/ask", response_model=RespuestaResponse)
def ask(body: PreguntaRequest) -> RespuestaResponse:
    try:
        contexto = recuperar(body.pregunta)
        respuesta = generar(body.pregunta, contexto)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    LLM_RESPUESTAS.inc()
    fuentes = [fuente for _, fuente in contexto]
    return RespuestaResponse(
        pregunta=body.pregunta, respuesta=respuesta, fuentes=fuentes
    )