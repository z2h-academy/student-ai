# Helmet API

API FastAPI para el proyecto RAG "librarian" del bootcamp AI Engineering.
Pone en produccion el pipeline RAG con observabilidad basica.

![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)

## Arquitectura

```
Client -> FastAPI (POST /api/ask) -> ChromaDB retriever -> Ollama/Mock responder -> Response
                                       |                      |
                                   ChromaDB               Ollama (local)
                                   (embeddings)           (LLM inference)
```

## Ejecutar con Docker

```bash
cp .env.example .env          # configurar variables
docker compose up --build     # levanta api + postgres
```

La API queda en `http://localhost:8000`.

## Ejecutar local (sin Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

## Endpoints

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| POST | `/api/ask` | Pregunta RAG. Body: `{"question": "...", "k": 5}` |
| GET | `/health` | Healthcheck. Retorna `status: ok` + version |
| GET | `/metrics` | Metricas en formato Prometheus |
| GET | `/metrics/json` | Metricas en JSON |

## Ejemplo de uso

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Que es un embedding?", "k": 3}'
```

Respuesta esperada:
```json
{
  "answer": "Respuesta simulada basada en ...",
  "sources": [{"content": "...", "metadata": {"source": "mock"}, "score": 0.9}],
  "model_used": "mock",
  "latency_ms": 12.5
}
```

## Cost Report

Genera un reporte de costos estimados por request:

```bash
python cost_report.py
```

Output: `cost_report.json` con desglose de tokens, latencia y costo USD.

## CI/CD

GitHub Actions ejecuta:
1. **Lint** — `ruff check`
2. **Test** — `pytest`
3. **Build + Push** — imagen Docker a GHCR (solo en `main`)

## Terraform

Template documentado en `terraform/` para ECS Fargate. No funcional sin adaptar VPC, subnets y security groups reales.

## Output esperado (Healthcheck)

```json
{"status": "ok", "version": "0.1.0"}
```

## Estructura

```
3-helmet/
├── api/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, endpoints, middleware
│   ├── models.py          # Pydantic request/response models
│   ├── metrics.py         # MetricsTracker (thread-safe, Prometheus)
│   └── dependencies.py    # DI: retriever, responder (singleton)
├── terraform/
│   ├── main.tf            # ECS task definition (template)
│   ├── variables.tf       # Variables configurables
│   └── outputs.tf         # Outputs: api_url, task_arn
├── .github/workflows/
│   └── ci.yml             # Lint + Test + Build/Push
├── Dockerfile             # Multi-stage build (3.12-slim)
├── .dockerignore
├── docker-compose.yml     # api + postgres
├── cost_report.py         # Calculo de costos por request
├── requirements.txt
├── .env.example
└── README.md
```
