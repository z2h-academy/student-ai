# level-4/labs/bench_inference.py
"""
Inferencia eficiente: mide latencia y tokens/segundo de llama3.2.

Ollama devuelve metricas reales en cada respuesta (prompt_eval_count,
eval_count, eval_duration): este lab las extrae y compara configuraciones.

Usage:
    python labs/bench_inference.py
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"
PREGUNTA = "Explica que es el RAG en 3 frases."


def generar(modelo: str, prompt_system: str, max_tokens: int) -> dict:
    """Una llamada a Ollama devolviendo texto + metricas del servidor."""
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": PREGUNTA},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }
    inicio = time.perf_counter()
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    latencia = time.perf_counter() - inicio

    datos = response.json()
    return {
        "texto": datos["message"]["content"],
        "latencia_s": round(latencia, 2),
        "tokens_salida": datos.get("eval_count", 0),
        "tokens_prompt": datos.get("prompt_eval_count", 0),
        "eval_s": round(datos.get("eval_duration", 1) / 1e9, 2),
        "tokens_por_segundo": round(
            datos.get("eval_count", 0)
            / max(datos.get("eval_duration", 1) / 1e9, 1e-9),
            1,
        ),
    }


def main() -> None:
    print("=== BENCH DE INFERENCIA (llama3.2 via Ollama) ===")
    print(f"Pregunta fija: {PREGUNTA!r}\n")

    configs = [
        ("max_tokens=64 ", "Eres un asistente util.", 64),
        ("max_tokens=128", "Eres un asistente util.", 128),
        ("system largo ", (
            "Eres un asistente experto en sistemas de IA que responde "
            "siempre con explicaciones tecnicas detalladas, incluyendo "
            "ejemplos de arquitectura y consideraciones de produccion."
        ), 128),
    ]

    print(f"{'config':<16} {'latencia':>9} {'tokens out':>11} {'tok/s':>7}")
    print("-" * 48)
    for nombre, system, max_tokens in configs:
        r = generar(MODELO, system, max_tokens)
        print(
            f"{nombre:<16} {r['latencia_s']:>7.2f}s {r['tokens_salida']:>11} "
            f"{r['tokens_por_segundo']:>7.1f}"
        )

    print("\nConclusiones:")
    print("  - El limite de salida (num_predict) domina la latencia:")
    print("    menos tokens permitidos = respuestas mas rapidas y baratas.")
    print("  - Un system prompt mas largo agrega tokens de entrada (costo)")
    print("    pero no cambia la velocidad de generacion por token.")


if __name__ == "__main__":
    main()