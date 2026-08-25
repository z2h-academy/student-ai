# level-3/labs/memory_state.py
"""
Memoria de corto plazo: el estado del agente dentro del loop.

El agente mantiene un historial de conversacion en su estado y lo
reenvia completo en cada turno (mismo patron que chat_history del
Level 1, pero ahora dentro de un agente con tools).

Usage:
    python labs/memory_state.py
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"


def get_hora() -> str:
    """Devuelve la hora actual."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_hora",
            "description": "Devuelve la hora actual en formato HH:MM:SS",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

EJECUTAR = {"get_hora": lambda a: get_hora()}

SYSTEM = (
    "Eres un asistente con memoria. Recuerdas la conversacion completa "
    "y respondes con naturalidad, usando el historial cuando sea util."
)


class AgenteConMemoria:
    """Agente con estado: historial de mensajes que persiste entre turnos."""

    def __init__(self) -> None:
        self.estado = {
            "messages": [{"role": "system", "content": SYSTEM}],
            "turnos": 0,
        }

    def chat(self, texto: str) -> str:
        """Un turno: actualiza el estado, resuelve tools y genera respuesta."""
        self.estado["messages"].append({"role": "user", "content": texto})
        self.estado["turnos"] += 1

        for _ in range(3):
            payload = {
                "model": MODELO,
                "messages": self.estado["messages"],
                "tools": TOOLS,
                "stream": False,
            }
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120
            )
            response.raise_for_status()
            mensaje = response.json()["message"]

            llamadas = mensaje.get("tool_calls", [])
            if not llamadas:
                self.estado["messages"].append(mensaje)
                return mensaje.get("content", "").strip()

            for llamada in llamadas:
                nombre = llamada["function"]["name"]
                argumentos = llamada["function"]["arguments"]
                if isinstance(argumentos, str):
                    argumentos = json.loads(argumentos or "{}")
                resultado = EJECUTAR[nombre](argumentos)
                self.estado["messages"].append(
                    {"role": "tool", "content": json.dumps({"resultado": resultado})}
                )

        return "No pude resolver el turno."


def main() -> None:
    agente = AgenteConMemoria()

    print("=== AGENTE CON MEMORIA (corto plazo) ===")
    preguntas = [
        "Me llamo Ana y trabajo en IA.",
        "Que hora es?",
        "Como me llamo y a que me dedico?",
    ]

    for pregunta in preguntas:
        print(f"\nTu> {pregunta}")
        print(f"Bot> {agente.chat(pregunta)}")
        print(f"[estado: {agente.estado['turnos']} turnos, "
              f"{len(agente.estado['messages'])} mensajes]")


if __name__ == "__main__":
    main()