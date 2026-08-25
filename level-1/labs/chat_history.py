# level-1/labs/chat_history.py
"""
Chat multi-turno: memoria de corto plazo con historial.

Usage:
    python labs/chat_history.py
"""

import requests

OLLAMA_HOST = "http://localhost:11434"


def chat(messages: list[dict], model: str = "llama3.2") -> str:
    """Envia la conversacion completa a Ollama y devuelve la respuesta."""
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
    historial: list[dict] = [
        {
            "role": "system",
            "content": (
                "Eres un asistente amigable en espanol. "
                "Recuerdas la conversacion y respondes con naturalidad."
            ),
        },
    ]

    print("=== CHAT MULTI-TURNO ===")
    print("(escribe 'salir' para terminar)")

    while True:
        usuario = input("\nTu> ").strip()
        if usuario.lower() in ("salir", "exit"):
            break

        historial.append({"role": "user", "content": usuario})
        respuesta = chat(historial)
        historial.append({"role": "assistant", "content": respuesta})

        print(f"Bot> {respuesta}")
        print(f"\n[historial: {len(historial)} mensajes]")

    print("\nFin del chat.")


if __name__ == "__main__":
    main()