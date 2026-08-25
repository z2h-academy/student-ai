# Proyecto 5 — `mentor`

## El especialista fine-tuneado para tu empresa

**Nivel:** Model Ops & Optimization · **Complejidad:** Avanzado
**Stack:** PEFT/QLoRA · vLLM · Ollama

Fine-tuning con **LoRA/QLoRA** del modelo de respuestas usando datos reales del
dominio (tickets y conversaciones del detective), servir el modelo con **vLLM**
y comparar contra el modelo base y contra prompt engineering.

---

## Estructura

```
5-mentor/
├── build_finetune_dataset.py   # state/ P4 + tickets → gold/finetune/
├── train_lora.py               # Fine-tuning LoRA (PEFT)
├── train_qlora.py              # Fine-tuning QLoRA (4-bit)
├── serve_vllm.py               # Servidor vLLM (API OpenAI-compatible)
├── bench_paths.py              # Comparativa: base vs prompt vs LoRA vs QLoRA
├── requirements.txt
├── .env.example
├── README.md
├── gold/finetune/              # Dataset de fine-tuning (generado)
├── lora_adapter/               # Adaptador LoRA (generado)
├── qlora_adapter/              # Adaptador QLoRA (generado)
├── lora_metrics.json           # Métricas LoRA
├── qlora_metrics.json          # Métricas QLoRA
├── bench_results.json          # Resultados benchmark
├── bench_results.csv           # Resultados benchmark (CSV)
└── notebooks/
    ├── 04-train-lora.ipynb     # Visualización fine-tuning LoRA
    └── 05-bench-paths.ipynb    # Comparativa de rutas
```

---

## Prerrequisitos

1. **Servicios levantados** (Docker Compose con perfil `l4`):
   ```bash
   docker compose -f ../ai-platform/ai.docker-compose.yml --profile l4 up -d
   ```

2. **Datos del Proyecto 4** en `../4-detective/state/` (o datos sintéticos de fallback)

3. **Datos del Proyecto 1** en `../1-triage/gold/amazon_reviews/`

---

## Paso 1 — Construir dataset de fine-tuning

```bash
python build_finetune_dataset.py
```

Genera `gold/finetune/dataset.jsonl` con formato (instruction, input, output).

---

## Paso 2 — Fine-tuning LoRA

```bash
python train_lora.py
```

Guarda adaptador en `./lora_adapter/` y métricas en `lora_metrics.json`.

---

## Paso 3 — Fine-tuning QLoRA

```bash
python train_qlora.py
```

Más eficiente en memoria (4-bit quantization). Guarda en `./qlora_adapter/`.

---

## Paso 4 — Servir con vLLM

```bash
# Con adaptador LoRA
python serve_vllm.py --adapter ./lora_adapter

# Con adaptador QLoRA
python serve_vllm.py --adapter ./qlora_adapter

# Modelo base (sin fine-tuning)
python serve_vllm.py
```

API compatible OpenAI en `http://localhost:8080`.

---

## Paso 5 — Benchmark comparativo

```bash
python bench_paths.py
```

Evalúa 4 rutas (base, prompt engineering, LoRA, QLoRA) y genera:
- `bench_results.json` — resultados detallados
- `bench_results.csv` — tabla comparativa

---

## Notebooks

```bash
jupyter lab notebooks/
```

- `04-train-lora.ipynb` — Visualiza el proceso de fine-tuning: loss curve, ejemplos antes/después
- `05-bench-paths.ipynb` — Comparativa de rutas: tablas, gráficos, decisión documentada

---

## Output esperado

| Archivo | Descripción |
|---------|-------------|
| `gold/finetune/dataset.jsonl` | Dataset de fine-tuning (instruction/input/output) |
| `lora_adapter/` | Adaptador LoRA entrenado |
| `qlora_adapter/` | Adaptador QLoRA entrenado |
| `lora_metrics.json` | Métricas de entrenamiento LoRA |
| `qlora_metrics.json` | Métricas de entrenamiento QLoRA |
| `bench_results.json` | Resultados benchmark (JSON) |
| `bench_results.csv` | Tabla comparativa (CSV) |
