# Proyecto 2 — Librarian (RAG)

El bibliotecario que sabe dónde está cada respuesta: un sistema RAG que
indexa la Knowledge Base del Proyecto 1 en ChromaDB, busca los chunks
más relevantes y genera respuestas con citas verificables.

## Flujo

```
KB chunks (P1)  →  index_kb.py  →  ChromaDB
                                     ↓
Pregunta seller  →  retriever.py  →  top-k chunks
                                     ↓
                              responder.py  →  Respuesta + citas
                                     ↓
                              eval_rag.py   →  Métricas de calidad
```

## Estructura

```
2-librarian/
├── index_kb.py          # Indexa chunks + embeddings en ChromaDB
├── retriever.py         # Búsqueda semántica (top-k chunks)
├── responder.py         # RAG completo: retrieval → LLM → respuesta + citas
├── eval_rag.py          # Evaluación: precision@k, recall@k, fidelidad
├── requirements.txt     # Dependencias
├── .env.example         # Template de variables de entorno
├── README.md            # Este archivo
└── notebooks/
    └── 02-eval-rag.ipynb  # Notebook de evaluación interactiva
```

## Requisitos

- Python 3.12+
- Parquets del Proyecto 1: `silver/kb/chunks.parquet` y `gold/embeddings/kb_chunks.parquet`
- Ollama corriendo en `localhost:11434` (para LLM local, por defecto)
- Opcional: API keys de OpenAI o Anthropic en `.env`

## Instalación

```bash
cd proyectos-ai/2-librarian
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus credenciales si usas OpenAI/Anthropic
```

## Uso

### 1. Indexar la KB

```bash
python index_kb.py --chunks <ruta>/chunks.parquet --embeddings <ruta>/kb_chunks.parquet
```

### 2. Hacer preguntas

```bash
python responder.py "¿Qué productos tienen mejor calificación?"
```

### 3. Evaluar el RAG

```bash
python eval_rag.py --model ollama --k 5
```

## Output esperado

### Indexación

```
============================================================
INDEXACIÓN COMPLETA | chunks=29,347 | dir=./chroma_db
============================================================
```

### Responder

```
============================================================
RESPUESTA:
============================================================
Según la base de conocimiento, los productos mejor calificados son...
(Fuente: B001E4KFG0) (Fuente: B0019553UC)
============================================================
```

### Evaluar

```json
{
  "total_questions": 15,
  "avg_precision_at_k": 0.82,
  "avg_recall_at_k": 0.75,
  "avg_fidelity": 0.94,
  "total_citations_correct": 58,
  "total_citations_incorrect": 2
}
```

## Modelos soportados

| Modelo | Comando | Costo | Requiere |
|--------|---------|-------|----------|
| Ollama (default) | `--model ollama` | Gratis | Ollama local |
| OpenAI | `--model openai` | ~$0.001/query | OPENAI_API_KEY |
| Anthropic | `--model anthropic` | ~$0.001/query | ANTHROPIC_API_KEY |

## Citation tracking

Cada respuesta incluye citas verificadas. El sistema:
1. Recupera los k chunks más relevantes
2. Pasa el contexto al LLM con instrucciones de citar
3. Valida que las fuentes citadas existan en los chunks recuperados
4. Reporta fidelidad de citas (correctas vs incorrectas vs sin citar)
