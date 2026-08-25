# Proyecto 4 — Detective: Sistema Multi-Agente con Herramientas y Memoria

El detective convierte la API de `helmet` (Proyecto 3) en un **sistema
multi-agente** que clasifica, resuelve y escala consultas de sellers usando
herramientas MCP y memoria persistente en PostgreSQL.

## Arquitectura

```
Consulta del seller
       │
       ▼
  ┌─────────────┐
  │  SUPERVISOR  │  LangGraph StateGraph
  │  (orquesta)  │
  └──────┬──────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ TRIAGE │─▶│ RESOLVER │
└────────┘ └────┬─────┘
                │
         ┌──────┼──────┐
         ▼      ▼      ▼
    ┌────────┐ ┌────────┐ ┌────────────┐
    │kb_search│ │responder│ │history_lookup│
    └────────┘ └────────┘ └────────────┘
                │
         (si confianza < umbral)
                │
                ▼
         ┌──────────┐
         │ESCALADOR │──▶ escalate_human
         └──────────┘
                │
                ▼
         Respuesta final
```

## Agentes

| Agente | Responsabilidad |
|--------|----------------|
| `TriageAgent` | Clasifica la consulta en categorias (facturacion, producto, logistica, general) con un LLM |
| `ResolverAgent` | Ejecuta herramientas (kb_search, responder, history_lookup) y genera respuesta con citas |
| `EscaladorAgent` | Genera ticket de escalacion cuando el resolver no puede o la confianza es baja |
| `SupervisorAgent` | Orquesta el flujo completo usando LangGraph StateGraph |

## Herramientas MCP

| Tool | Descripcion | Backend |
|------|-------------|---------|
| `kb_search` | Busca chunks relevantes en la knowledge base | ChromaDB (P2) |
| `responder` | Genera respuestas con contexto | API de helmet (P3) o Ollama |
| `history_lookup` | Consulta historial de tickets del seller | PostgreSQL |
| `escalate_human` | Registra ticket de escalacion | PostgreSQL + MinIO S3 |

## Memoria

- **Conversacion:** tabla `conversations` en PostgreSQL (seller_id, role, content, timestamp)
- **Estado de agentes:** JSON en `s3://gold/amazon_reviews/state/`

## Requisitos

- Python 3.12+
- Docker Compose con perfil `l3` (Ollama + PostgreSQL + MinIO)
- ChromaDB con la KB indexada (Proyecto 2)

## Instalacion

```bash
cd proyectos-ai/4-detective
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus credenciales
```

## Ejecucion

```bash
# Verificar que los servicios estan corriendo
docker compose -f ai-platform/ai.docker-compose.yml --profile l3 ps

# Ejecutar el notebook demo
jupyter notebook notebooks/03-demo-agente.ipynb

# O usar directamente
python -c "
from agents.supervisor import SupervisorAgent
s = SupervisorAgent()
result = s.run('Me cobraron el doble en mi factura')
print(result['final_output'])
"
```

## Estructura

```
4-detective/
├── agents/
│   ├── __init__.py
│   ├── triage.py          # Clasificacion de consultas
│   ├── resolver.py        # Resolucion con herramientas
│   ├── escalador.py       # Escalamiento a humano
│   └── supervisor.py      # Orquestador con LangGraph
├── tools/
│   ├── __init__.py
│   ├── kb_search.py       # Busqueda en ChromaDB
│   ├── responder.py       # Generacion de respuestas
│   ├── history_lookup.py  # Historial en PostgreSQL
│   └── escalate_human.py  # Registro de escalaciones
├── memory/
│   ├── __init__.py
│   └── memory.py          # Memoria de conversacion (PG)
├── state/
│   ├── __init__.py
│   └── state.py           # Estado del agente (S3)
├── notebooks/
│   └── 03-demo-agente.ipynb
├── requirements.txt
├── .env.example
└── README.md
```

## Flujo del Supervisor (LangGraph)

El `SupervisorAgent` construye un `StateGraph` con los siguientes nodos y aristas:

1. **triage** → clasifica la consulta
2. **resolver** → ejecuta herramientas y genera respuesta
3. **(condicional)** → si necesita escalamiento o confianza < umbral → **escalador**
4. **format_answer** → formatea la respuesta final
5. **END**

El estado se comparte entre nodos via `AgentState` (TypedDict).
