# level-3/labs/react_agent.py
"""
Bucle ReAct manual: el agente piensa, actua (usa una tool), observa y repite.

Usage:
    python labs/react_agent.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"
MAX_PASOS = 5


def get_hora() -> str:
    """Devuelve la hora actual."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def capital(texto: str) -> str:
    """Devuelve el texto en mayusculas."""
    return texto.upper()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_hora",
            "description": "Devuelve la hora actual en formato HH:MM:SS",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
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
]

EJECUTAR = {
    "get_hora": lambda a: get_hora(),
    "capital": lambda a: capital(a["texto"]),
}


def paso_react(historial: list[dict]) -> dict:
    """Un paso del loop: pide al modelo su siguiente accion."""
    payload = {
        "model": MODELO,
        "messages": historial,
        "tools": TOOLS,
        "stream": False,
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


def resolver(pregunta: str) -> str:
    """Loop ReAct: razona, usa tools y observa hasta llegar a la respuesta."""
    historial = [{"role": "user", "content": pregunta}]

    for paso in range(MAX_PASOS):
        mensaje = paso_react(historial)
        historial.append(mensaje)

        contenido = mensaje.get("content", "")
        llamadas = mensaje.get("tool_calls", [])

        if contenido:
            print(f"[paso {paso + 1}] razonamiento: {contenido.strip()}")

        if not llamadas:
            return contenido.strip()

        for llamada in llamadas:
            nombre = llamada["function"]["name"]
            argumentos = llamada["function"]["arguments"]
            if isinstance(argumentos, str):
                argumentos = json.loads(argumentos or "{}")
            resultado = EJECUTAR[nombre](argumentos)
            print(f"[paso {paso + 1}] tool {nombre}{argumentos} -> {resultado!r}")
            historial.append(
                {"role": "tool", "content": json.dumps({"resultado": resultado})}
            )

    return "No llegue a una respuesta en el maximo de pasos."


if __name__ == "__main__":
    print("=== AGENTE REACT ===")
    print("Pregunta: Que hora es?")
    print("Respuesta:", resolver("Que hora es?"))
    print()
    print("Pregunta: Escribe la palabra hola en mayusculas")
    print("Respuesta:", resolver("Escribe la palabra hola en mayusculas"))