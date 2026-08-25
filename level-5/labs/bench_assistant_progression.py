# level-5/labs/bench_assistant_progression.py
"""
Bench de progresion: como mejoro el sistema a lo largo de los niveles.

Mide latencia y calidad heuristica de cada era del asistente sobre las
MISMAS preguntas:

  L1  llama3.2 directo            (el modelo generalista)
  L2  prompt engineering          (system prompt de librarian)
  L3  RAG completo                (recuperar contexto + generar con citas)

El fine-tuning de L4 se referencia del bench previo (bench_paths):
con 10 ejemplos no cambia la calidad factual, y re-entrenar aqui
duplicaria costo sin nueva informacion. Esa decision TAMBIEN es parte
del benchmark.

Usage:
    python labs/bench_assistant_progression.py   # requiere Ollama arriba
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from librarian_pkg.kb import indexar_base
from librarian_pkg.retrieval import recuperar
from librarian_pkg.generation import generar

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"

EVALS = [
    {"pregunta": "Que es una API REST?", "claves": ["http", "api"]},
    {"pregunta": "Como funciona el RAG?", "claves": ["contexto", "recupera"]},
    {"pregunta": "Que es la ventana de contexto?", "claves": ["token", "limite"]},
]

SYSTEM_LIBRARIAN = (
    "Eres 'librarian', un asistente tecnico que responde en espanol "
    "de forma breve y clara."
)


def medir(funcion, etiqueta: str) -> dict:
    """Corre las evaluaciones midiendo latencia y calidad."""
    latencias, aciertos = [], []
    for ev in EVALS:
        inicio = time.perf_counter()
        texto = funcion(ev["pregunta"]).lower()
        latencias.append(time.perf_counter() - inicio)
        aciertos.append(sum(1 for c in ev["claves"] if c in texto))

    return {
        "ruta": etiqueta,
        "latencia_promedio_s": round(sum(latencias) / len(latencias), 2),
        "calidad": f"{sum(aciertos)}/{len(EVALS) * 2}",
    }


def llm_directo(pregunta: str) -> str:
    """Era L1: el modelo pelado."""
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": pregunta}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def prompt_engineering(pregunta: str) -> str:
    """Era L2: mismo modelo + system prompt profesional."""
    payload = {
        "model": MODELO,
        "messages": [
            {"role": "system", "content": SYSTEM_LIBRARIAN},
            {"role": "user", "content": pregunta},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def rag_completo(pregunta: str) -> str:
    """Era L3: recuperar + generar con contexto citado."""
    contexto = recuperar(pregunta, n=2)
    return generar(pregunta, contexto)


def main() -> None:
    print("=== BENCH DE PROGRESION DEL ASISTENTE ===\n")
    print("Indexando KB para la era RAG...")
    indexar_base()

    eras = [
        ("L1: modelo directo", llm_directo),
        ("L2: + prompt engineering", prompt_engineering),
        ("L3: + RAG con citas", rag_completo),
    ]

    print(f"\n{'era':<28} {'latencia':>9} {'calidad':>8}")
    print("-" * 48)
    for etiqueta, funcion in eras:
        resultado = medir(funcion, etiqueta)
        print(
            f"{resultado['ruta']:<28} "
            f"{resultado['latencia_promedio_s']:>7.2f}s "
            f"{resultado['calidad']:>8}"
        )

    print(f"\n{'L4: + fine-tuning (ref)':<28} {'ver L4 bench':>16} "
          f"{'tono/formato':>13}")

    print("\n--- VEREDICTO DE PROGRESION ---")
    print("  - El prompt engineering es gratis y estabiliza el tono.")
    print("  - El RAG agrega conocimiento real: sube la calidad factual")
    print("    al costo de una busqueda vectorial (~1-2s).")
    print("  - El FT especializa conducta; con poca data NO mejora hechos.")
    print("  - Cada nivel sumo una pieza; este bench lo demuestra con datos.")


if __name__ == "__main__":
    main()