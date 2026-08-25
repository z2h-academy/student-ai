# Proyectos AI Engineering — Z2H-Shop Assistant

Seis proyectos encadenados que construyen un asistente de IA completo para sellers de e-commerce.

## Cadena de proyectos

| # | Proyecto | Descripción | Nivel |
|---|----------|-------------|-------|
| 1 | **triage** | ETL medallion + clasificador de urgencia | Fundamentos |
| 2 | **librarian** | RAG con ChromaDB + citations | LLM Engineering |
| 3 | **helmet** | API FastAPI + Docker + CI/CD | Deployment |
| 4 | **detective** | Agentes multi-agente + memoria | Agentic AI |
| 5 | **mentor** | Fine-tuning LoRA/QLoRA + benchmark | Model Ops |
| 6 | **mission-control** | Observability + Airflow + K8s | Infra & Production |

## Flujo de datos

```
P1 (triage) → P2 (librarian) → P3 (helmet) → P4 (detective) → P5 (mentor) → P6 (mission-control)
     ↓               ↓               ↓               ↓               ↓               ↓
  MinIO Bronze    ChromaDB        FastAPI API     Multi-agente    Fine-tuned      Sistema
  → Silver        → Retrieval     → Docker        → Memory        → vLLM          completo
  → Gold          → Citations     → CI/CD         → State         → Benchmark
```

## Stack por proyecto

- **P1:** pandas, pyarrow, scikit-learn, sentence-transformers, MinIO
- **P2:** chromadb, sentence-transformers, OpenAI/Anthropic/Ollama
- **P3:** FastAPI, Docker, GitHub Actions, Terraform
- **P4:** LangGraph, MCP, PostgreSQL, MinIO
- **P5:** PEFT, bitsandbytes, vLLM, transformers
- **P6:** Prometheus, Airflow, Kubernetes, guardrails

## Ejecución

Cada proyecto es independiente pero consume los outputs del anterior. Ver `README.md` de cada carpeta para instrucciones específicas.

```bash
# Ejemplo: ejecutar P1
cd 1-triage
pip install -r requirements.txt
python etl_pipeline/main.py --step all
python triage_classifier.py
```

## Variables de entorno

Cada proyecto tiene un `.env.example` con las variables requeridas. Copia a `.env` y completa con tus credenciales.

## Notebooks

| Proyecto | Notebook | Contenido |
|----------|----------|-----------|
| P1 | `01_exploracion.ipynb` | EDA del dataset Amazon Reviews |
| P2 | `02-eval-rag.ipynb` | Evaluación de retrieval y citations |
| P4 | `03-demo-agente.ipynb` | Demo del sistema multi-agente |
| P5 | `04-train-lora.ipynb` | Proceso de fine-tuning LoRA |
| P5 | `05-bench-paths.ipynb` | Benchmark: base vs prompt vs LoRA vs QLoRA |
