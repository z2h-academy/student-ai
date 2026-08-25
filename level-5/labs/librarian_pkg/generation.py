# level-2/labs/librarian_pkg/generation.py
"""Generacion: construye la respuesta con el LLM usando el contexto."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO_CHAT = "llama3.2"


def generar(pregunta: str, contexto: list[tuple[str, str]]) -> str:
    """Genera una respuesta con citas de fuente usando el contexto recuperado."""
    lineas = [f"[Fuente: {fuente}] {parrafo}" for parrafo, fuente in contexto]
    contexto_texto = "\n\n".join(lineas)

    system = (
        "Eres 'librarian', un asistente que responde usando SOLO el contexto. "
        "Al final de cada dato citado, indica la fuente entre parentesis "
        "usando el formato (Fuente: NOMBRE). "
        "Si el contexto no tiene la respuesta, di 'No tengo informacion sobre eso'."
    )
    user = f"Contexto:\n{contexto_texto}\n\nPregunta: {pregunta}"

    payload = {
        "model": MODELO_CHAT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]
