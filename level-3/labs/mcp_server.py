# level-3/labs/mcp_server.py
"""
Servidor MCP: expone herramientas del agente via Model Context Protocol.

Las tools se registran con @mcp.tool() y el servidor se ejecuta en
modo stdio (comunicacion por stdin/stdout con el cliente).

Usage:
    python labs/mcp_server.py
"""

import os
from pathlib import Path

from mcp.server import MCPServer

mcp = MCPServer("librarian-tools")


@mcp.tool()
def get_hora() -> str:
    """Devuelve la hora actual en formato HH:MM:SS."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


@mcp.tool()
def kb_search(pregunta: str) -> str:
    """Busca parrafos relevantes a la pregunta en la base de conocimiento."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from librarian_pkg.kb import indexar_base
    from librarian_pkg.retrieval import recuperar

    ruta_kb = "data/knowledge_base.md"
    indexar_base(ruta_kb)

    resultados = recuperar(pregunta, n=2)
    return "\n\n".join(f"[Fuente: {fuente}] {parrafo}" for parrafo, fuente in resultados)


if __name__ == "__main__":
    mcp.run()