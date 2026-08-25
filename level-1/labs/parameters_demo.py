# level-1/labs/parameters_demo.py
"""
Parametros del modelo: temperatura, top_p y max_tokens.

Usage:
    python labs/parameters_demo.py
"""

import requests

OLLAMA_HOST = "http://localhost:11434"


def chat(
    messages: list[dict],
    model: str = "llama3.2",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int | None = None,
) -> str:
    """Envia un chat con parametros configurables."""
    url = f"{OLLAMA_HOST}/api/chat"
    options: dict = {"temperature": temperature, "top_p": top_p}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> None:
    prompt = [
        {
            "role": "user",
            "content": (
                "Inventa una historia corta de 2-3 frases "
                "sobre un robot que aprende a cocinar."
            ),
        }
    ]

    print("=== TEMPERATURA: misma pregunta, 3 valores ===")
    for temp in (0.0, 0.7, 1.3):
        print(f"\n--- temperature={temp} ---")
        print(chat(prompt, temperature=temp))

    print()
    print("=" * 60)
    print("=== MAX_TOKENS: limite de respuesta ===")
    print("\n--- sin limite ---")
    respuesta_larga = chat(
        [
            {
                "role": "user",
                "content": "Explica que es el machine learning en detalle.",
            }
        ],
        temperature=0.3,
    )
    print(respuesta_larga)
    print(f"\n[longitud del texto: {len(respuesta_larga)} caracteres]")

    print("\n--- con max_tokens=30 ---")
    respuesta_corta = chat(
        [
            {
                "role": "user",
                "content": "Explica que es el machine learning en detalle.",
            }
        ],
        temperature=0.3,
        max_tokens=30,
    )
    print(respuesta_corta)
    print(f"\n[longitud del texto: {len(respuesta_corta)} caracteres]")


if __name__ == "__main__":
    main()