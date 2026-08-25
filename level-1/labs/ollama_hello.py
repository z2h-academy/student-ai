# level-1/labs/ollama_hello.py
"""
Primer contacto con Ollama desde Python.

Usage:
    python labs/ollama_hello.py
"""

import json

import requests

OLLAMA_HOST = "http://localhost:11434"


def generate(prompt: str, model: str = "llama3.2") -> str:
    """Envia un prompt a Ollama y devuelve la respuesta como texto."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["response"]


def main() -> None:
    prompt = "Dime en una frase que es un LLM"
    print(f"Prompt: {prompt}")
    print("-" * 50)
    answer = generate(prompt)
    print(f"Respuesta: {answer}")


if __name__ == "__main__":
    main()