# level-3/labs/tool_calling.py
"""
Tool calling nativo de Ollama: el LLM decide que herramienta usar.

Usage:
    python labs/tool_calling.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"


def get_hora() -> str:
    """Tool 1: devuelve la hora actual."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def suma(a: int, b: int) -> int:
    """Tool 2: suma dos numeros enteros."""
    return a + b


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
            "name": "suma",
            "description": "Suma dos numeros enteros",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    },
]

EJECUTAR = {
    "get_hora": lambda args: get_hora(),
    "suma": lambda args: suma(args["a"], args["b"]),
}


def chat_con_tools(pregunta: str) -> None:
    """Envia la pregunta con tools y ejecuta las llamadas que el modelo decida."""
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": pregunta}],
        "tools": TOOLS,
        "stream": False,
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    mensaje = response.json()["message"]

    print(f"Contenido del modelo: {mensaje.get('content', '')!r}")
    print(f"Tool calls pedidos: {mensaje.get('tool_calls')}")

    for llamada in mensaje.get("tool_calls", []):
        nombre = llamada["function"]["name"]
        argumentos = llamada["function"]["arguments"]
        if isinstance(argumentos, str):
            argumentos = json.loads(argumentos or "{}")
        resultado = EJECUTAR[nombre](argumentos)
        print(f"-> Ejecutando {nombre}{argumentos}: {resultado}")


if __name__ == "__main__":
    chat_con_tools("Que hora es?")
    print()
    chat_con_tools("Cuanto es 17 + 25?")