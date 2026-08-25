# level-3/labs/tools.py
"""
Tools propias: el agente reutiliza el RAG de Level 2 como herramientas.

Reutiliza librarian_pkg (copiado de level-2/labs/) para exponer:
    kb_search(pregunta)  -> parrafos relevantes con fuentes (ChromaDB)
    responder(pregunta)  -> respuesta generada con el contexto

Usage:
    python labs/tools.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian_pkg.kb import indexar_base
from librarian_pkg.retrieval import recuperar
from librarian_pkg.generation import generar

RUTA_KB = "data/knowledge_base.md"


def kb_search(pregunta: str, n: int = 2) -> list[dict]:
    """Tool: busca los parrafos mas relevantes a la pregunta en la KB."""
    resultados = recuperar(pregunta, n=n)
    return [{"parrafo": p, "fuente": f} for p, f in resultados]


def responder(pregunta: str) -> str:
    """Tool: genera una respuesta con citas usando el contexto recuperado."""
    contexto = recuperar(pregunta, n=2)
    return generar(pregunta, contexto)


TOOLS = {
    "kb_search": {
        "descripcion": "Busca parrafos relevantes a una pregunta en la base de conocimiento",
        "funcion": kb_search,
        "schema": {
            "type": "function",
            "function": {
                "name": "kb_search",
                "description": "Busca parrafos relevantes a una pregunta en la base de conocimiento",
                "parameters": {
                    "type": "object",
                    "properties": {"pregunta": {"type": "string"}},
                    "required": ["pregunta"],
                },
            },
        },
    },
    "responder": {
        "descripcion": "Genera una respuesta con citas usando el contexto de la KB",
        "funcion": responder,
        "schema": {
            "type": "function",
            "function": {
                "name": "responder",
                "description": "Genera una respuesta con citas usando el contexto de la KB",
                "parameters": {
                    "type": "object",
                    "properties": {"pregunta": {"type": "string"}},
                    "required": ["pregunta"],
                },
            },
        },
    },
}


def listar_tools() -> None:
    """Muestra el registro de herramientas disponibles."""
    print("=== HERRAMIENTAS REGISTRADAS ===")
    for nombre, tool in TOOLS.items():
        print(f"  - {nombre}: {tool['descripcion']}")


def ejecutar_tool(nombre: str, **kwargs) -> object:
    """Ejecuta una herramienta por nombre."""
    if nombre not in TOOLS:
        raise ValueError(f"Tool desconocida: {nombre}")
    return TOOLS[nombre]["funcion"](**kwargs)


if __name__ == "__main__":
    print(f"Indexando KB ({RUTA_KB})...")
    indexar_base(RUTA_KB)
    print()

    listar_tools()
    print()

    print("=== PRUEBA DE TOOLS ===")
    resultado = ejecutar_tool("kb_search", pregunta="Que es una API REST?")
    print(f"kb_search -> {resultado[0]['parrafo'][:70]}... (fuente: {resultado[0]['fuente']})")
    print()
    respuesta = ejecutar_tool("responder", pregunta="Que es una API REST?")
    print(f"responder -> {respuesta[:120]}...")