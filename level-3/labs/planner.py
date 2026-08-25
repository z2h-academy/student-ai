# level-3/labs/planner.py
"""
Planning & reflexion: el agente planifica, ejecuta y se autocritica.

  1. PLAN: el agente descompone la tarea en pasos
  2. EJECUCION: resuelve cada paso (con tools si las necesita)
  3. REFLEXION: se autocritica y mejora la respuesta

Usage:
    python labs/planner.py
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"


def llamar(prompt: str, system: str = "Eres un asistente util.") -> str:
    """Una llamada de chat simple a Ollama."""
    payload = {
        "model": MODELO,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def planificar(tarea: str) -> str:
    """Paso 1: descompone la tarea en pasos ordenados."""
    return llamar(
        f"Descompone esta tarea en 3 pasos concretos y ordenados:\n{tarea}\n"
        "Formato: '1. ...\\n2. ...\\n3. ...'",
        system="Eres un planificador. Devuelves pasos cortos y accionables.",
    )


def ejecutar_paso(paso: str) -> str:
    """Paso 2: resuelve un paso individual."""
    return llamar(
        f"Resuelve este paso de forma breve y concreta:\n{paso}",
        system="Eres un ejecutor. Responde directo y sin rodeos.",
    )


def reflexionar(tarea: str, borrador: str) -> str:
    """Paso 3: se autocritica y devuelve la version final mejorada."""
    critica = llamar(
        f"Tarea: {tarea}\n\nBorrador: {borrador}\n\n"
        "Que le falta a este borrador? Que se puede mejorar? "
        "Responde en 2 frases.",
        system="Eres un critico riguroso.",
    )
    print(f"\nCRITICA: {critica}")

    final = llamar(
        f"Tarea: {tarea}\n\nBorrador: {borrador}\n\n"
        f"Critica: {critica}\n\n"
        "Reescribe la respuesta final incorporando la critica.",
        system="Eres un editor que mejora borradores.",
    )
    return final


def main() -> None:
    tarea = "Explica que es una API REST y por que importa en 3 frases."

    print("=== AGENTE CON PLANIFICACION Y REFLEXION ===\n")
    print(f"TAREA: {tarea}\n")

    print("--- 1. PLAN ---")
    plan = planificar(tarea)
    print(plan)

    print("\n--- 2. EJECUCION (paso a paso) ---")
    borrador = ""
    for linea in plan.splitlines():
        if linea.strip():
            resultado = ejecutar_paso(linea.strip())
            print(f"  -> {resultado}")
            borrador += resultado + " "

    print("\n--- 3. REFLEXION ---")
    final = reflexionar(tarea, borrador)
    print(f"\nRESPUESTA FINAL: {final}")


if __name__ == "__main__":
    main()