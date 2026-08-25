"""Agente escalador: genera tickets de escalamiento cuando el resolver no puede."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from tools.escalate_human import escalate

load_dotenv()

ESCALATOR_SYSTEM_PROMPT = """\
Eres un agente de escalamiento para un asistente de sellers de e-commerce.
Cuando la resolución automatizada no puede atender la consulta, generas un ticket
de escalamiento con todo el contexto necesario para un agente humano.

Recibirás:
- La consulta original del seller
- La categoría detectada (triage)
- La razón por la que se necesita escalamiento
- Los resultados de las herramientas ejecutadas

Genera un ticket de escalamiento con un JSON valido:
{
    "seller_id": "<id del seller>",
    "subject": "<asunto breve del ticket>",
    "description": "<descripcion detallada>",
    "priority": "<low|medium|high|urgent>",
    "category": "<categoria del triage>",
    "context_summary": "<resumen del contexto para el agente humano>"
}
"""


class EscaladorAgent:
    """Genera y registra tickets de escalamiento."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.getenv("MODEL_NAME", "llama3.2")
        self._base_url = base_url or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self._llm = ChatOllama(model=self._model, base_url=self._base_url, temperature=0)

    def escalate(
        self,
        query: str,
        categoria: str,
        razon_escalamiento: str,
        seller_id: str = "default",
        tool_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Genera el ticket y lo registra via la tool escalate_human."""
        messages = [
            SystemMessage(content=ESCALATOR_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "consulta": query,
                "categoria": categoria,
                "razon_escalamiento": razon_escalamiento,
                "seller_id": seller_id,
                "tool_results_keys": list(tool_results.keys()) if tool_results else [],
            }, ensure_ascii=False)),
        ]

        response = self._llm.invoke(messages)
        raw = response.content.strip()

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            parsed = {
                "seller_id": seller_id,
                "subject": f"Escalamiento automatico - {categoria}",
                "description": query,
                "priority": "medium",
                "category": categoria,
                "context_summary": razon_escalamiento,
            }

        parsed["seller_id"] = parsed.get("seller_id", seller_id)

        ticket_id = escalate(parsed)

        return {
            "ticket_id": ticket_id,
            "ticket_data": parsed,
        }
