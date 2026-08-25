"""Agente de triaje: clasifica la consulta del seller en categorias."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

CATEGORIES = ["facturacion", "producto", "logistica", "general"]

TRIAGE_SYSTEM_PROMPT = """\
Eres un agente de triaje para un asistente de sellers de e-commerce.
Clasifica la consulta del seller en UNA de estas categorias:
- facturacion: cobros, facturas, devoluciones de dinero, cargos, pagos
- producto: caracteristicas del producto, disponibilidad, variantes, especificaciones
- logistica: envios, entregas, tiempos de entrega, tracking, devoluciones fisicas
- general: todo lo demas (cuentas, configuracion, quejas generales)

Responde UNICAMENTE con un JSON valido:
{"categoria": "<una_de_las_categorias>", "confianza": <float_entre_0_y_1>, "razon": "<breve explicacion>"}

Ejemplos:
Consulta: "Me cobraron el doble en mi factura del mes pasado"
Respuesta: {"categoria": "facturacion", "confianza": 0.95, "razon": "El seller menciona un cobro duplicado en factura"}

Consulta: "El producto que recibí no tiene las características que prometía"
Respuesta: {"categoria": "producto", "confianza": 0.90, "razon": "El seller reporta desajuste entre producto recibido y descripcion"}

Consulta: "Mi pedido lleva 2 semanas sin llegar"
Respuesta: {"categoria": "logistica", "confianza": 0.92, "razon": "El seller reporta retraso en entrega de pedido"}

Consulta: "Como cambio el correo de mi cuenta?"
Respuesta: {"categoria": "general", "confianza": 0.88, "razon": "Solicitud de configuracion de cuenta"}
"""


class TriageAgent:
    """Clasifica consultas del seller usando un LLM."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model or os.getenv("MODEL_NAME", "llama3.2")
        self._base_url = base_url or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        self._llm = ChatOllama(model=self._model, base_url=self._base_url, temperature=0)

    def classify(self, query: str) -> dict[str, object]:
        """Clasifica una consulta y retorna categoria + confianza + razon."""
        messages = [
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        response = self._llm.invoke(messages)
        raw = response.content.strip()

        try:
            parsed = json.loads(raw)
            categoria = parsed.get("categoria", "general")
            if categoria not in CATEGORIES:
                categoria = "general"
            return {
                "categoria": categoria,
                "confianza": float(parsed.get("confianza", 0.5)),
                "razon": parsed.get("razon", "Clasificacion por defecto"),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "categoria": "general",
                "confianza": 0.3,
                "razon": f"No se pudo parsear respuesta del LLM: {raw[:200]}",
            }
