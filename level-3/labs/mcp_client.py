# level-3/labs/mcp_client.py
"""
Cliente MCP: consume las tools de un servidor MCP por stdio.

Levanta el servidor mcp_server.py como subproceso, lista sus tools
y llama a una de ellas.

Usage:
    python labs/mcp_client.py
"""

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

RUTA_SERVIDOR = str(Path(__file__).resolve().parent / "mcp_server.py")


async def main() -> None:
    parametros = StdioServerParameters(
        command=sys.executable,
        args=[RUTA_SERVIDOR],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    async with stdio_client(parametros) as (lectura, escritura):
        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()

            print("=== TOOLS DEL SERVIDOR MCP ===")
            tools = await sesion.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            print()
            print("=== LLAMANDO get_hora ===")
            resultado = await sesion.call_tool("get_hora", {})
            print(f"  Resultado: {resultado.content[0].text}")

            print()
            print("=== LLAMANDO kb_search ===")
            resultado = await sesion.call_tool(
                "kb_search", {"pregunta": "Que es una API REST?"}
            )
            texto = resultado.content[0].text
            print(f"  Resultado (primeros 100 chars): {texto[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())