# level-1/labs/cloud_openai.py
"""
Integracion con APIs OpenAI-compatible (OpenCode Go, OpenAI, Ollama).

El mismo codigo funciona con cualquier proveedor que hable el protocolo
OpenAI-compatible: solo cambia la URL base, la key y el modelo en .env.

Usage:
    python labs/cloud_openai.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuracion del proveedor (se lee de .env)
# OPENAI_API_KEY admite fallback a OPENCODE_GO_API_KEY para simplificar el setup
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENCODE_GO_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat_openai(prompt: str, system: str = "Eres un asistente util.") -> str:
    """Envia un chat al proveedor OpenAI-compatible y devuelve la respuesta."""
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content or ""


def main() -> None:
    print(f"Proveedor: {OPENAI_BASE_URL} | Modelo: {OPENAI_MODEL}")
    respuesta = chat_openai("Explica que es una API en 2 oraciones.")
    print("Respuesta:")
    print(respuesta)


if __name__ == "__main__":
    main()
