from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

from .dependencies import get_responder, get_retriever
from .metrics import metrics
from .models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetricsResponse,
    Source,
)

logger = logging.getLogger("helmet")

APP_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Helmet API starting — version %s", APP_VERSION)
    yield
    logger.info("Helmet API shutting down")


app = FastAPI(
    title="Helmet API",
    description="RAG librarian API — produccion con observabilidad basica",
    version=APP_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    log_entry = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": elapsed_ms,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    prometheus_text = metrics.to_prometheus()
    return PlainTextResponse(content=prometheus_text, media_type="text/plain")


@app.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json() -> MetricsResponse:
    return MetricsResponse(**metrics.to_dict())


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    start = time.perf_counter()

    retriever = get_retriever()
    responder = get_responder()

    query_results = retriever.query(request.question, n_results=request.k)  # type: ignore[union-attr]

    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]

    context_parts: list[str] = []
    sources: list[Source] = []
    for i, doc in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        score = round(1.0 - distances[i], 4) if i < len(distances) else 0.0
        context_parts.append(doc)
        sources.append(
            Source(
                content=doc[:200],
                metadata=meta if isinstance(meta, dict) else {},
                score=score,
            )
        )

    context_text = "\n---\n".join(context_parts)
    answer, tokens = responder.generate(request.question, context=context_text)  # type: ignore[union-attr]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    metrics.record_request(latency_ms=elapsed_ms, tokens=tokens)

    model_used = "mock" if "Mock" in type(responder).__name__ else "ollama"

    return AskResponse(
        answer=answer,
        sources=sources,
        model_used=model_used,
        latency_ms=elapsed_ms,
    )


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
