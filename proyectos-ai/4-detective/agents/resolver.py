"""Agente resolver: ejecuta herramientas para responder la consulta del seller."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from tools.kb_search import kb_search
from tools.responder import responder
from tools.history_lookup import history_lookup

load_dotenv()

RESOLVER_SYSTEM_PROMPT = """\
Eres un agente de resolución para un asistente de sellers de e-commerce.
Tu trabajo es responder la consulta del seller usando la información disponible.

Tienes acceso a estas herramientas (ya ejecutadas previamente):
- kb_search: búsqueda en la knowledge base de productos
- responder: generación de respuesta con el LLM base
- history_lookup: historial de tickets anteriores del seller

Recibirás:
1. La categoría de la consulta (triage)
2. La consulta original del seller
3. Los resultados de las herramientas ejecutadas

Genera una respuesta final que:
- Responda directamente la consulta
- Incluya citas o referencias a la fuente cuando aplique
- Sea clara y profesional
- Indique si necesita escalación a un humano

Responde con un JSON valido:
{"respuesta": "<texto de la respuesta>", "citas": ["<fuente1>", ...], "necesita_escalamiento": <bool>, "razon_escalamiento": "<razon si aplica>"}
"""


class ResolverAgent:
    """Resuelve consultas usando herramientas MCP y un LLM."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.getenv("MODEL_NAME", "llama3.2")
        self._base_url = base_url or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self._llm = ChatOllama(model=self._model, base_url=self._base_url, temperature=0.3)

    def resolve(
        self,
        query: str,
        categoria: str,
        seller_id: str = "default",
        k: int = 5,
    ) -> dict[str, Any]:
        """Ejecuta herramientas y genera una respuesta final."""
        tool_results: dict[str, Any] = {}

        kb_results = kb_search(query, k=k)
        tool_results["kb_search"] = kb_results

        context_parts = []
        for chunk in kb_results:
            context_parts.append(chunk.get("content", ""))
        context = "\n---\n".join(context_parts) if context_parts else "Sin contexto disponible"

        answer, tokens = responder(query, context=context)
        tool_results["responder"] = {"answer": answer, "tokens": tokens}

        history = history_lookup(seller_id)
        tool_results["history_lookup"] = history

        history_summary = ""
        if history:
            recent = history[:3]
            history_summary = f"\nHistorial reciente ({len(history)} tickets):\n"
            for ticket in recent:
                history_summary += f"  - [{ticket.get('status', 'unknown')}] {ticket.get('subject', 'sin asunto')}\n"

        messages = [
            SystemMessage(content=RESOLVER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "categoria": categoria,
                "consulta": query,
                "contexto_kb": context[:1000],
                "respuesta_responder": answer,
                "historial": history_summary,
            }, ensure_ascii=False)),
        ]

        response = self._llm.invoke(messages)
        raw = response.content.strip()

        try:
            parsed = json.loads(raw)
            return {
                "respuesta": parsed.get("respuesta", answer),
                "citas": parsed.get("citas", []),
                "necesita_escalamiento": parsed.get("necesita_escalamiento", False),
                "razon_escalamiento": parsed.get("razon_escalamiento", ""),
                "tool_results": tool_results,
                "tokens_used": tokens,
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "respuesta": answer,
                "citas": [],
                "necesita_escalamiento": False,
                "razon_escalamiento": "",
                "tool_results": tool_results,
                "tokens_used": tokens,
            }
