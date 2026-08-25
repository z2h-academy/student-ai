# Project 1 — Triage: Amazon Reviews Urgency Classifier

## Descripción

Clasificador de urgencia de reseñas de Amazon utilizando un pipeline ETL medallion (Bronze → Silver → Gold) y un modelo de regresión logística sobre embeddings pre-computados.

### Arquitectura Medallion

```
Bronze (raw)  →  Silver (clean + splits + KB)  →  Gold (embeddings)
     ↓                    ↓                              ↓
 Reviews.csv      reviews.parquet              embeddings.parquet
                 train/test.parquet           KB chunks embeddings
                 KB chunks.parquet
```

### Niveles de urgencia

| Urgencia | Score | Significado |
|----------|-------|-------------|
| 1 | 1 estrella | Crítica — atención inmediata |
| 2 | 2 estrellas | Negativa — experiencia mala |
| 3 | 3 estrellas | Neutral — promedio |
| 4 | 4 estrellas | Positiva — buena |
| 5 | 5 estrellas | Muy positiva — excelente |

## Requisitos

- Python 3.12+
- Docker Compose con servicios MinIO y PostgreSQL levantados
- Credenciales en `.env` (ver `.env.example`)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar todo el pipeline

```bash
python -m etl_pipeline.main --step all
```

### Ejecutar un paso específico

```bash
python -m etl_pipeline.main --step bronze    # Descargar datos
python -m etl_pipeline.main --step silver    # Limpiar + splits + KB
python -m etl_pipeline.main --step gold      # Generar embeddings
```

### Entrenar el clasificador

```bash
python triage_classifier.py
```

### Evaluar el modelo

```bash
python eval_triage.py
```

### Notebook de exploración

```bash
jupyter lab notebooks/01_exploracion.ipynb
```

## Output esperado

Al ejecutar el pipeline completo, se generan:

- `silver/clean/reviews.parquet` — ~394K reviews limpias
- `silver/splits/train.parquet` — 80% de datos (estratificado)
- `silver/splits/test.parquet` — 20% de datos (estratificado)
- `silver/kb/chunks.parquet` — chunks de reviews para KB
- `gold/amazon_reviews/embeddings/kb_chunks.parquet` — embeddings de KB
- `gold/amazon_reviews/embeddings/train_embeddings.parquet` — embeddings de train
- `gold/amazon_reviews/embeddings/test_embeddings.parquet` — embeddings de test
- `state/model.joblib` — modelo entrenado
- `state/classification_report.json` — métricas de evaluación
- `state/confusion_matrix.json` — matriz de confusión

## Estructura del proyecto

```
1-triage/
├── etl_pipeline/
│   ├── __init__.py
│   ├── config.yml
│   ├── utils.py
│   ├── main.py
│   ├── bronze/
│   │   ├── __init__.py
│   │   └── extract.py
│   ├── silver/
│   │   ├── __init__.py
│   │   ├── transform_silver.py
│   │   ├── split_data.py
│   │   └── build_kb.py
│   └── gold/
│       ├── __init__.py
│       └── build_embeddings.py
├── notebooks/
│   └── 01_exploracion.ipynb
├── state/
├── triage_classifier.py
├── eval_triage.py
├── requirements.txt
├── .env.example
└── README.md
```
