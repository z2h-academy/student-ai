"""Agente supervisor: orquesta triage → resolver → (opcional) escalador con LangGraph."""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.triage import TriageAgent
from agents.resolver import ResolverAgent
from agents.escalador import EscaladorAgent
from memory.memory import ConversationMemory

load_dotenv()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))


class AgentState(TypedDict):
    """Estado del grafo de agentes."""
    query: str
    seller_id: str
    categoria: str
    confianza_triaje: float
    razon_triaje: str
    respuesta: str
    citas: list[str]
    necesita_escalamiento: bool
    razon_escalamiento: str
    ticket_id: str
    ticket_data: dict[str, Any]
    tool_results: dict[str, Any]
    tokens_used: int
    final_output: str


class SupervisorAgent:
    """Orquesta el flujo triage → resolver → escalador usando LangGraph."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._triage = TriageAgent(model=model, base_url=base_url)
        self._resolver = ResolverAgent(model=model, base_url=base_url)
        self._escalador = EscaladorAgent(model=model, base_url=base_url)
        self._memory = ConversationMemory()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("triage", self._triage_node)
        graph.add_node("resolver", self._resolver_node)
        graph.add_node("escalador", self._escalador_node)
        graph.add_node("format_answer", self._format_answer_node)

        graph.set_entry_point("triage")
        graph.add_edge("triage", "resolver")
        graph.add_conditional_edges(
            "resolver",
            self._should_escalate,
            {
                "escalador": "escalador",
                "format_answer": "format_answer",
            },
        )
        graph.add_edge("escalador", "format_answer")
        graph.add_edge("format_answer", END)

        return graph.compile()

    def _triage_node(self, state: AgentState) -> dict[str, Any]:
        result = self._triage.classify(state["query"])
        return {
            "categoria": result["categoria"],
            "confianza_triaje": result["confianza"],
            "razon_triaje": result["razon"],
        }

    def _resolver_node(self, state: AgentState) -> dict[str, Any]:
        result = self._resolver.resolve(
            query=state["query"],
            categoria=state["categoria"],
            seller_id=state.get("seller_id", "default"),
        )
        return {
            "respuesta": result["respuesta"],
            "citas": result["citas"],
            "necesita_escalamiento": result["necesita_escalamiento"],
            "razon_escalamiento": result["razon_escalamiento"],
            "tool_results": result.get("tool_results", {}),
            "tokens_used": result.get("tokens_used", 0),
        }

    def _should_escalate(self, state: AgentState) -> str:
        if state.get("necesita_escalamiento", False):
            return "escalador"
        if state.get("confianza_triaje", 1.0) < CONFIDENCE_THRESHOLD:
            return "escalador"
        return "format_answer"

    def _escalador_node(self, state: AgentState) -> dict[str, Any]:
        result = self._escalador.escalate(
            query=state["query"],
            categoria=state["categoria"],
            razon_escalamiento=state.get("razon_escalamiento", "Baja confianza del sistema"),
            seller_id=state.get("seller_id", "default"),
            tool_results=state.get("tool_results", {}),
        )
        return {
            "ticket_id": result["ticket_id"],
            "ticket_data": result["ticket_data"],
        }

    def _format_answer_node(self, state: AgentState) -> dict[str, Any]:
        ticket_id = state.get("ticket_id", "")

        if ticket_id:
            final = (
                f"[ESCALADO - Ticket {ticket_id}]\n"
                f"Categoria: {state.get('categoria', 'desconocida')}\n"
                f"Asunto: {state.get('ticket_data', {}).get('subject', '')}\n"
                f"Un agente humano revisara tu caso pronto."
            )
        else:
            citations = state.get("citas", [])
            citation_text = ""
            if citations:
                citation_text = "\n\nFuentes:\n" + "\n".join(
                    f"  - {c}" for c in citations
                )
            final = f"{state.get('respuesta', '')}{citation_text}"

        return {"final_output": final}

    def run(
        self,
        query: str,
        seller_id: str = "default",
    ) -> dict[str, Any]:
        """Ejecuta el flujo completo del supervisor y retorna el resultado."""
        initial_state: AgentState = {
            "query": query,
            "seller_id": seller_id,
            "categoria": "",
            "confianza_triaje": 0.0,
            "razon_triaje": "",
            "respuesta": "",
            "citas": [],
            "necesita_escalamiento": False,
            "razon_escalamiento": "",
            "ticket_id": "",
            "ticket_data": {},
            "tool_results": {},
            "tokens_used": 0,
            "final_output": "",
        }

        result = self._graph.invoke(initial_state)

        self._memory.save_message(seller_id, "user", query)
        self._memory.save_message(seller_id, "assistant", result.get("final_output", ""))

        return {
            "final_output": result.get("final_output", ""),
            "categoria": result.get("categoria", ""),
            "confianza_triaje": result.get("confianza_triaje", 0.0),
            "ticket_id": result.get("ticket_id", ""),
            "tokens_used": result.get("tokens_used", 0),
        }

    def get_history(self, seller_id: str) -> list[dict[str, str]]:
        """Recupera el historial de conversacion de un seller."""
        return self._memory.get_conversation(seller_id)
