# deploy_librarian — Puesta en produccion

Este es el cierre del nivel: el sistema `librarian` completo, operado como servicio.

## Que incluye

```
deploy_librarian/
├── README.md            ← este archivo (documentacion de despliegue)
└── (referencia a labs/)
    ├── librarian_pkg/   ← el RAG como paquete (seccion 1)
    ├── api_main.py      ← la API FastAPI con Pydantic y observabilidad
    ├── models.py        ← request/response tipados
    ├── metrics.py       ← logs + metricas
    ├── cost_report.py   ← costo por request
    ├── Dockerfile       ← imagen reproducible
    ├── docker-compose.yml ← orquestacion
    ├── test_api.py      ← tests automaticos
    └── .github/workflows/ci.yml ← CI/CD
```

## Como levantar (en el Codespace)

```bash
# 1. Dependencias
pip install -r level-2/requirements.txt

# 2. Copiar la KB (si no esta)
mkdir -p level-2/data && cp level-1/labs/knowledge_base.md level-2/data/

# 3. En desarrollo (con recarga)
cd level-2 && uvicorn labs.api_main:app --reload

# 4. En produccion (con Docker Compose)
cd level-2/labs && docker compose up -d --build
```

## Endpoints

| Metodo | Ruta | Que hace |
|--------|------|----------|
| GET | `/health` | Verifica que el servicio esta vivo |
| GET | `/metrics` | Resumen de requests, latencia y errores |
| POST | `/api/index` | Indexa la base de conocimiento |
| POST | `/api/ask` | Responde con citas y fuentes |
| GET | `/docs` | Documentacion interactiva (OpenAPI) |

## Como probar

```bash
# Tests automaticos
cd level-2 && python -m pytest labs/test_api.py -v

# Request manual
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Que es una API REST?"}'

# Costo por request
cd level-2 && python -m labs.cost_report
```

## CI/CD

El workflow `.github/workflows/ci.yml` corre los tests en cada push a `main`.
Se copia a la raiz del repo:

```bash
cp level-2/labs/.github/workflows/ci.yml .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "ci: add test pipeline"
git push origin main
```

## Checklist de cierre del nivel

- [ ] La API responde `/health`, `/metrics`, `/api/ask`
- [ ] `docker compose up -d --build` levanta el servicio con Ollama
- [ ] `pytest` pasa (5 tests verdes)
- [ ] El CI corre en GitHub Actions al hacer push
- [ ] `/metrics` muestra latencia por endpoint
- [ ] `cost_report` muestra el costo USD por request
