# level-1/labs/prompt_basics.py
"""
Prompt basico: comparacion entre sin system prompt y con system prompt.

Usage:
    python labs/prompt_basics.py
"""

import requests

OLLAMA_HOST = "http://localhost:11434"


def chat(messages: list[dict], model: str = "llama3.2") -> str:
    """Envia una conversacion completa a Ollama y devuelve la respuesta."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> None:
    pregunta = "Explica que es una API en maximo 2 oraciones."

    # Caso 1: sin system prompt (solo el mensaje del usuario)
    sin_system = [
        {"role": "user", "content": pregunta},
    ]

    # Caso 2: con system prompt (reglas de comportamiento)
    con_system = [
        {
            "role": "system",
            "content": (
                "Eres un asistente tecnico en espanol. "
                "Respondes siempre en espanol, en maximo 2 oraciones, "
                "con lenguaje simple y sin jerga tecnica."
            ),
        },
        {"role": "user", "content": pregunta},
    ]

    print("=== CASO 1: SIN system prompt ===")
    print(chat(sin_system))
    print()
    print("=== CASO 2: CON system prompt ===")
    print(chat(con_system))


if __name__ == "__main__":
    main()