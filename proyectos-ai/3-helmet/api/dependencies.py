from __future__ import annotations

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from collections.abc import Callable

_responder_instance: Callable | None = None
_retriever_instance: object | None = None


def get_retriever() -> object:
    """Singleton retriever backed by ChromaDB (or mock if unavailable)."""
    global _retriever_instance  # noqa: PLW0603
    if _retriever_instance is not None:
        return _retriever_instance

    try:
        import chromadb

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name="librarian_docs",
            metadata={"hnsw:space": "cosine"},
        )
        _retriever_instance = collection
    except Exception:
        _retriever_instance = _MockRetriever()

    return _retriever_instance


def get_responder() -> Callable:
    """Singleton responder. Uses Ollama if available, else mock."""
    global _responder_instance  # noqa: PLW0603
    if _responder_instance is not None:
        return _responder_instance

    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "")
    model_name = os.getenv("MODEL_NAME", "llama3.2")

    if ollama_endpoint:
        _responder_instance = _OllamaResponder(ollama_endpoint, model_name)
    else:
        _responder_instance = _MockResponder()

    return _responder_instance


class _MockRetriever:
    """Simula un retriever para standalone testing."""

    def query(self, query_text: str, n_results: int = 5) -> dict:
        return {
            "documents": [[f"Documento simulado para: {query_text}"]],
            "metadatas": [[{"source": "mock", "page": "1"}]],
            "distances": [[0.1]],
        }


class _MockResponder:
    """Simula un LLM para standalone testing."""

    def generate(self, prompt: str, context: str = "") -> tuple[str, int]:
        answer = (
            f"Respuesta simulada basada en {len(context)} caracteres de contexto. "
            "Configura OLLAMA_ENDPOINT para usar un modelo real."
        )
        return answer, len(answer.split())


class _OllamaResponder:
    """Wrapper sobre la API de Ollama."""

    def __init__(self, endpoint: str, model: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model

    def generate(self, prompt: str, context: str = "") -> tuple[str, int]:
        import requests

        full_prompt = f"Contexto:\n{context}\n\nPregunta: {prompt}\n\nRespuesta:"
        resp = requests.post(
            f"{self._endpoint}/api/generate",
            json={"model": self._model, "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "")
        tokens = data.get("eval_count", len(answer.split()))
        return answer, tokens
