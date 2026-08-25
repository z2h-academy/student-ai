from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pregunta a responder")
    k: int = Field(default=5, ge=1, le=20, description="Numero de fuentes a recuperar")


class Source(BaseModel):
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)
    score: float = Field(default=0.0)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    avg_tokens: int
