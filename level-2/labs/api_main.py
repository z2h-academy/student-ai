# level-2/labs/api_main.py
"""API FastAPI del servicio librarian con modelos Pydantic y observabilidad.

Usage:
    uvicorn labs.api_main:app --reload
"""

import logging

from fastapi import FastAPI, HTTPException

from labs.librarian_pkg import indexar_base, recuperar, generar
from labs.models import PreguntaRequest, RespuestaResponse
from labs.metrics import registrar_metricas, metricas_resumen

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Librarian API",
    description="Asistente RAG que responde con citas de la base de conocimiento.",
    version="0.3.0",
)

registrar_metricas(app)


@app.get("/health")
def health() -> dict:
    """Verifica que el servicio esta vivo."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    """Resumen de metricas: requests, latencia y errores."""
    return metricas_resumen()


@app.post("/api/index")
def api_index() -> dict:
    """Indexa la base de conocimiento en ChromaDB."""
    n = indexar_base()
    return {"indexados": n}


@app.post("/api/ask", response_model=RespuestaResponse)
def ask(body: PreguntaRequest) -> RespuestaResponse:
    """Responde una pregunta usando el RAG y devuelve las fuentes."""
    try:
        contexto = recuperar(body.pregunta)
        respuesta = generar(body.pregunta, contexto)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    fuentes = [fuente for _, fuente in contexto]
    return RespuestaResponse(pregunta=body.pregunta, respuesta=respuesta, fuentes=fuentes)
