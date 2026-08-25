# level-3/labs/multi_agent.py
"""
Multi-agente: supervisor + workers orquestados desde cero.

El SUPERVISOR recibe la consulta y decide que WORKER la resuelve.
Cada worker es una funcion especializada que usa sus propias tools.
El supervisor compone la respuesta final del worker elegido.

Usage:
    python labs/multi_agent.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"


def get_hora() -> str:
    """Tool del worker de tiempo."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def capital(texto: str) -> str:
    """Tool del worker de texto."""
    return texto.upper()


WORKERS = {
    "tiempo": {
        "descripcion": "Resuelve preguntas sobre la hora actual",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_hora",
                    "description": "Devuelve la hora actual",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
        "ejecutar": {"get_hora": lambda a: get_hora()},
    },
    "texto": {
        "descripcion": "Convierte textos a mayusculas",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "capital",
                    "description": "Convierte un texto a mayusculas",
                    "parameters": {
                        "type": "object",
                        "properties": {"texto": {"type": "string"}},
                        "required": ["texto"],
                    },
                },
            },
        ],
        "ejecutar": {"capital": lambda a: capital(a["texto"])},
    },
}


def supervisor(consulta: str) -> str:
    """Clasifica la consulta y devuelve el nombre del worker indicado."""
    prompt = (
        "Clasifica la siguiente consulta en UNO de estos workers: "
        f"{', '.join(WORKERS.keys())}.\n"
        "Ejemplos:\n"
        "- 'Que hora es?' -> tiempo\n"
        "- 'Dame la hora actual' -> tiempo\n"
        "- 'Convierte hola a mayusculas' -> texto\n"
        "- 'Escribe esto en MAYUSCULAS' -> texto\n"
        f"Consulta: {consulta}\n"
        "Responde SOLO con el nombre del worker, sin explicaciones."
    )
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    nombre = response.json()["message"]["content"].strip().lower()

    if nombre not in WORKERS:
        nombre = "tiempo"
    return nombre


def worker(nombre: str, consulta: str) -> str:
    """Ejecuta la consulta con las tools del worker elegido (loop ReAct corto)."""
    config = WORKERS[nombre]
    historial = [{"role": "user", "content": consulta}]

    for paso in range(3):
        payload = {
            "model": MODELO,
            "messages": historial,
            "tools": config["tools"],
            "stream": False,
        }
        response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        mensaje = response.json()["message"]
        historial.append(mensaje)

        llamadas = mensaje.get("tool_calls", [])
        if not llamadas:
            return mensaje.get("content", "").strip()

        for llamada in llamadas:
            nombre_tool = llamada["function"]["name"]
            argumentos = llamada["function"]["arguments"]
            if isinstance(argumentos, str):
                argumentos = json.loads(argumentos or "{}")
            resultado = config["ejecutar"][nombre_tool](argumentos)
            historial.append(
                {"role": "tool", "content": json.dumps({"resultado": resultado})}
            )

    return "El worker no llego a una respuesta."


def orquestar(consulta: str) -> None:
    """Flujo completo: supervisor clasifica, worker resuelve."""
    print(f"Consulta: {consulta}")

    elegido = supervisor(consulta)
    print(f"SUPERVISOR -> worker '{elegido}'")

    respuesta = worker(elegido, consulta)
    print(f"WORKER '{elegido}' -> {respuesta}")
    print()


if __name__ == "__main__":
    print("=== MULTI-AGENTE: SUPERVISOR + WORKERS ===\n")
    orquestar("Que hora es?")
    orquestar("Escribe la palabra hola en mayusculas")