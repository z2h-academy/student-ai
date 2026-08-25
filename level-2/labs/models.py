# level-2/labs/models.py
"""Modelos Pydantic: definen el contrato de la API (request y response)."""

from pydantic import BaseModel, Field


class PreguntaRequest(BaseModel):
    """Cuerpo de la peticion POST /api/ask."""

    pregunta: str = Field(..., min_length=1, max_length=500, description="La pregunta a responder")


class RespuestaResponse(BaseModel):
    """Respuesta del endpoint POST /api/ask."""

    pregunta: str
    respuesta: str
    fuentes: list[str] = []
