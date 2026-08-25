# level-2/labs/librarian_pkg/__init__.py
"""
librarian_pkg — el asistente RAG de Level 1 reestructurado como paquete.

Expone la API de alto nivel del RAG:
    indexar_base()  -> indexa la KB en ChromaDB
    preguntar(q)    -> recupera contexto y genera respuesta con citas

Usage (desde level-2/):
    from labs.librarian_pkg import indexar_base, preguntar
    indexar_base()
    print(preguntar("¿Que es una API REST?"))
"""

from .kb import indexar_base, RUTA_KB
from .retrieval import recuperar
from .generation import generar

__all__ = ["indexar_base", "recuperar", "generar", "RUTA_KB"]


def preguntar(pregunta: str, n: int = 3) -> str:
    """Flujo completo: recuperar contexto y generar respuesta con citas."""
    contexto = recuperar(pregunta, n=n)
    return generar(pregunta, contexto)
