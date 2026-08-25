"""Tool MCP: genera respuestas usando la API de helmet (P3) o LLM local."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def responder(query: str, context: str = "") -> tuple[str, int]:
    """Genera una respuesta basada en la consulta y el contexto.

    Intenta usar la API de helmet (P3) si HELMET_API_URL esta configurado,
    de lo contrario usa Ollama directamente.

    Args:
        query: Pregunta del seller.
        context: Contexto recuperado de la KB.

    Returns:
        Tupla de (respuesta, tokens_utilizados).
    """
    helmet_url = os.getenv("HELMET_API_URL", "")
    if helmet_url:
        return _call_helmet_api(query, helmet_url)

    return _call_ollama(query, context)


def _call_helmet_api(query: str, base_url: str) -> tuple[str, int]:
    """Llama a la API de helmet (P3) para generar la respuesta."""
    import requests

    try:
        url = f"{base_url.rstrip('/')}/api/ask"
        resp = requests.post(
            url,
            json={"question": query, "k": 5},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        tokens = data.get("tokens_used", len(answer.split()))
        return answer, tokens
    except Exception:
        return _call_ollama(query, "")


def _call_ollama(query: str, context: str) -> tuple[str, int]:
    """Llama a Ollama directamente para generar la respuesta."""
    import requests

    base_url = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    model = os.getenv("MODEL_NAME", "llama3.2")

    prompt = f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta:" if context else f"Pregunta: {query}\n\nRespuesta:"

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "")
        tokens = data.get("eval_count", len(answer.split()))
        return answer, tokens
    except Exception:
        return (
            "No se pudo generar la respuesta. Verifica que Ollama este corriendo.",
            0,
        )
