# level-1/labs/embeddings_demo.py
"""
Embeddings: de texto a vectores y similitud coseno.

Usage:
    python labs/embeddings_demo.py
"""

import numpy as np
import requests

OLLAMA_HOST = "http://localhost:11434"
MODELO_EMBEDDING = "nomic-embed-text"


def obtener_embedding(texto: str) -> list[float]:
    """Devuelve el vector (embedding) de un texto usando Ollama."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": MODELO_EMBEDDING, "prompt": texto}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["embedding"]


def similitud_coseno(a: list[float], b: list[float]) -> float:
    """Mide la similitud entre dos vectores: 1 = iguales, 0 = sin relacion."""
    va = np.array(a)
    vb = np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main() -> None:
    frases = [
        "El gato duerme en el sofa",
        "Un gato descansa sobre el sofa",
        "Hoy hace mucho calor en la ciudad",
        "El clima esta muy caluroso hoy",
    ]

    print(f"=== EMBEDDINGS con {MODELO_EMBEDDING} ===")
    vectores = {}
    for frase in frases:
        vectores[frase] = obtener_embedding(frase)
        print(f"[{len(vectores[frase])} dimensiones] {frase}")

    print()
    print("=== SIMILITUD COSENO entre pares ===")
    pares = [
        (frases[0], frases[1]),
        (frases[0], frases[2]),
        (frases[2], frases[3]),
        (frases[0], frases[3]),
    ]
    for a, b in pares:
        sim = similitud_coseno(vectores[a], vectores[b])
        print(f"{sim:.3f} | {a}")
        print(f"        | {b}")
        print()


if __name__ == "__main__":
    main()