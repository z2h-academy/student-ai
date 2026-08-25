# level-1/labs/cloud_anthropic.py
"""
Integracion con la API de Anthropic (Claude) — eleccion del alumno.

A diferencia de OpenAI/OpenCode Go/Ollama, Anthropic NO usa el patron
OpenAI-compatible: usa su propio SDK y su propio formato de respuesta.

Usage:
    python labs/cloud_anthropic.py
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")


def chat_claude(prompt: str, system: str = "Eres un asistente util.") -> str:
    """Envia un chat a Claude y devuelve la respuesta."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.content[0].text


def main() -> None:
    print(f"Modelo: {ANTHROPIC_MODEL}")
    respuesta = chat_claude("Explica que es una API en 2 oraciones.")
    print("Respuesta de Claude:")
    print(respuesta)


if __name__ == "__main__":
    main()
