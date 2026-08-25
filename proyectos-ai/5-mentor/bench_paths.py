"""
bench_paths.py
Comparativa: modelo base vs prompt engineering vs LoRA vs QLoRA.

Métricas:
  - Calidad: evaluador LLM (score 1-5)
  - Costo entrenamiento: GPU-horas, tiempo, pérdida
  - Costo inferencia: latencia, tokens/s

Genera tabla comparativa en JSON + CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
METRICS_DIR = BASE_DIR
LORA_METRICS = METRICS_DIR / "lora_metrics.json"
QLORA_METRICS = METRICS_DIR / "qlora_metrics.json"
OUTPUT_JSON = BASE_DIR / "bench_results.json"
OUTPUT_CSV = BASE_DIR / "bench_results.csv"

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
MODEL_BASE = os.getenv("MODEL_BASE", "microsoft/phi-2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_BASE)

# Prompts de evaluación
SYSTEM_PROMPT = (
    "Eres un evaluador experto de asistentes de soporte para e-commerce. "
    "Evalúa la calidad de la respuesta en una escala del 1 al 5."
)

EVAL_TEMPLATE = (
    "Evalúa la siguiente respuesta de un agente de soporte.\n\n"
    "Pregunta del seller: {query}\n"
    "Respuesta: {response}\n\n"
    "Criterios:\n"
    "5: Perfecta — resuelve el problema, es empática, profesional.\n"
    "4: Buena — resuelve la mayoría del problema.\n"
    "3: Aceptable — parcialmente útil.\n"
    "2: Mala — no resuelve o es incorrecta.\n"
    "1: Muy mala — irrelevante o dañina.\n\n"
    "Responde SOLO con un número del 1 al 5."
)

# Preguntas de test
TEST_QUERIES: list[dict[str, str]] = [
    {
        "query": "Mi cliente recibió el producto dañado y quiere un reembolso urgente",
        "context": "Pedido #9876: funda de laptop, daño en esquina del empaque",
    },
    {
        "query": "¿Cómo puedo mejorar el ranking de mis productos en la categoría electrónica?",
        "context": "Vendedor con 3 meses de experiencia, 15 productos activos",
    },
    {
        "query": "El envío de mi último pedido se retrasó 5 días y el cliente está furioso",
        "context": "Envío estándar, destino: Lima,-tracking sin actualizar",
    },
    {
        "query": "¿Qué hago si un cliente deja una reseña falsa de 1 estrella?",
        "context": "Producto con 4.5 estrellas promedio, 200+ reseñas",
    },
    {
        "query": "Necesito configurar envíos internacionales a Europa pero no sé por dónde empezar",
        "context": "Vendedor nuevo en categoría moda, sin experiencia en exportación",
    },
]

# Prompt engineering
PROMPT_ENGINEERING_TEMPLATE = (
    "Eres el asistente de soporte del Z2H-Shop, una tienda online de tecnología "
    "y accesorios. Tu objetivo es ayudar a los sellers con problemas operativos, "
    "de logística y de atención al cliente.\n\n"
    "Responde SIEMPRE:\n"
    "1. Con empatía y profesionalismo\n"
    "2. Con pasos concretos y accionables\n"
    "3. Ofreciendo alternativas cuando sea posible\n"
    "4. Mencionando políticas de la tienda cuando aplique\n\n"
    "Consulta del seller: {query}\n"
    "Contexto adicional: {context}\n\n"
    "Respuesta:"
)


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str | None = None) -> tuple[str, float]:
    """Llama a Ollama y retorna (respuesta, tiempo_segundos)."""
    import httpx

    model = model or OLLAMA_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 256},
    }

    start = time.time()
    try:
        response = httpx.post(
            f"{OLLAMA_ENDPOINT}/api/generate",
            json=payload,
            timeout=120.0,
        )
        elapsed = time.time() - start
        response.raise_for_status()
        return response.json().get("response", ""), elapsed
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  [WARN] Error en Ollama: {exc}")
        return f"[Error: {exc}]", elapsed


def call_vllm(prompt: str, port: int = 8080) -> tuple[str, float]:
    """Llama a vLLM (OpenAI-compatible) y retorna (respuesta, tiempo_segundos)."""
    import httpx

    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7,
    }

    start = time.time()
    try:
        response = httpx.post(
            f"http://localhost:{port}/v1/completions",
            json=payload,
            timeout=120.0,
        )
        elapsed = time.time() - start
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if choices:
            return choices[0].get("text", ""), elapsed
        return "", elapsed
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  [WARN] Error en vLLM: {exc}")
        return f"[Error: {exc}]", elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_response(
    query: str,
    response: str,
    evaluator_model: str | None = None,
) -> float:
    """Evalúa una respuesta usando el evaluador LLM."""
    eval_prompt = EVAL_TEMPLATE.format(query=query, response=response)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{eval_prompt}"

    score_text, _ = call_ollama(full_prompt, evaluator_model)

    # Extraer número del 1-5
    for char in score_text.strip():
        if char.isdigit() and 1 <= int(char) <= 5:
            return float(char)
    return 3.0  # Default si no se puede parsear


# ---------------------------------------------------------------------------
# Bench runners
# ---------------------------------------------------------------------------

def bench_base(queries: list[dict[str, str]]) -> dict[str, Any]:
    """Evalúa el modelo base sin fine-tuning ni prompt engineering."""
    print("\n  [Base] Modelo base sin modificaciones")
    results: list[dict[str, Any]] = []
    total_latency = 0.0

    for q in queries:
        response, latency = call_ollama(q["query"])
        total_latency += latency
        score = evaluate_response(q["query"], response)
        results.append({
            "query": q["query"],
            "response": response,
            "score": score,
            "latency_seconds": latency,
        })
        print(f"    Score: {score}, Latency: {latency:.2f}s")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0

    return {
        "path": "base",
        "description": "Modelo base (sin FT, sin prompt engineering)",
        "avg_quality_score": avg_score,
        "avg_latency_seconds": avg_latency,
        "total_latency_seconds": total_latency,
        "num_queries": len(results),
        "details": results,
    }


def bench_prompt_engineering(queries: list[dict[str, str]]) -> dict[str, Any]:
    """Evalúa con prompt engineering mejorado."""
    print("\n  [Prompt] Prompt engineering")
    results: list[dict[str, Any]] = []
    total_latency = 0.0

    for q in queries:
        enhanced_prompt = PROMPT_ENGINEERING_TEMPLATE.format(
            query=q["query"],
            context=q["context"],
        )
        response, latency = call_ollama(enhanced_prompt)
        total_latency += latency
        score = evaluate_response(q["query"], response)
        results.append({
            "query": q["query"],
            "response": response,
            "score": score,
            "latency_seconds": latency,
        })
        print(f"    Score: {score}, Latency: {latency:.2f}s")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0

    return {
        "path": "prompt_engineering",
        "description": "Prompt engineering (instrucciones + ejemplos)",
        "avg_quality_score": avg_score,
        "avg_latency_seconds": avg_latency,
        "total_latency_seconds": total_latency,
        "num_queries": len(results),
        "details": results,
    }


def bench_lora(queries: list[dict[str, str]], port: int = 8080) -> dict[str, Any]:
    """Evalúa el modelo fine-tuneado con LoRA via vLLM."""
    print("\n  [LoRA] Fine-tuning LoRA via vLLM")
    results: list[dict[str, Any]] = []
    total_latency = 0.0

    for q in queries:
        response, latency = call_vllm(q["query"], port=port)
        total_latency += latency
        score = evaluate_response(q["query"], response)
        results.append({
            "query": q["query"],
            "response": response,
            "score": score,
            "latency_seconds": latency,
        })
        print(f"    Score: {score}, Latency: {latency:.2f}s")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0

    # Cargar métricas de entrenamiento si existen
    train_metrics: dict[str, Any] = {}
    if LORA_METRICS.exists():
        with open(LORA_METRICS, encoding="utf-8") as fh:
            train_metrics = json.load(fh)

    return {
        "path": "lora",
        "description": "Fine-tuning LoRA (r=16, alpha=32)",
        "avg_quality_score": avg_score,
        "avg_latency_seconds": avg_latency,
        "total_latency_seconds": total_latency,
        "num_queries": len(results),
        "train_loss": train_metrics.get("train_loss"),
        "train_time_seconds": train_metrics.get("train_runtime_seconds"),
        "details": results,
    }


def bench_qlora(queries: list[dict[str, str]], port: int = 8080) -> dict[str, Any]:
    """Evalúa el modelo fine-tuneado con QLoRA via vLLM."""
    print("\n  [QLoRA] Fine-tuning QLoRA via vLLM")
    results: list[dict[str, Any]] = []
    total_latency = 0.0

    for q in queries:
        response, latency = call_vllm(q["query"], port=port)
        total_latency += latency
        score = evaluate_response(q["query"], response)
        results.append({
            "query": q["query"],
            "response": response,
            "score": score,
            "latency_seconds": latency,
        })
        print(f"    Score: {score}, Latency: {latency:.2f}s")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    avg_latency = total_latency / len(results) if results else 0

    # Cargar métricas de entrenamiento si existen
    train_metrics: dict[str, Any] = {}
    if QLORA_METRICS.exists():
        with open(QLORA_METRICS, encoding="utf-8") as fh:
            train_metrics = json.load(fh)

    return {
        "path": "qlora",
        "description": "Fine-tuning QLoRA (4-bit, r=16, alpha=32)",
        "avg_quality_score": avg_score,
        "avg_latency_seconds": avg_latency,
        "total_latency_seconds": total_latency,
        "num_queries": len(results),
        "train_loss": train_metrics.get("train_loss"),
        "train_time_seconds": train_metrics.get("train_runtime_seconds"),
        "quantization": train_metrics.get("quantization"),
        "details": results,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_comparison_table(all_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Construye una tabla comparativa de todas las rutas."""
    rows: list[dict[str, Any]] = []
    for result in all_results:
        rows.append({
            "Ruta": result["path"],
            "Descripción": result["description"],
            "Calidad (avg)": result["avg_quality_score"],
            "Latencia (s)": result["avg_latency_seconds"],
            "Loss entrenamiento": result.get("train_loss", "N/A"),
            "Tiempo entrenamiento (s)": result.get("train_time_seconds", "N/A"),
            "Consultas evaluadas": result["num_queries"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Calidad (avg)", ascending=False, kind="stable")
    return df


def save_results(
    all_results: list[dict[str, Any]],
    df: pd.DataFrame,
) -> None:
    """Guarda resultados en JSON y CSV."""
    # JSON
    output = {
        "benchmark_results": all_results,
        "comparison_summary": df.to_dict(orient="records"),
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    # CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n  Resultados guardados:")
    print(f"    JSON: {OUTPUT_JSON}")
    print(f"    CSV:  {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta el benchmark completo de las 4 rutas."""
    parser = argparse.ArgumentParser(description="Benchmark de rutas de calidad")
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8080,
        help="Puerto de vLLM (default: 8080)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=["base", "prompt", "lora", "qlora"],
        default=["base", "prompt", "lora", "qlora"],
        help="Rutas a evaluar (default: todas)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("BENCHMARK — Base vs Prompt vs LoRA vs QLoRA")
    print("=" * 60)

    all_results: list[dict[str, Any]] = []

    if "base" in args.paths:
        print("\n[1/4] Evaluando modelo base...")
        all_results.append(bench_base(TEST_QUERIES))

    if "prompt" in args.paths:
        print("\n[2/4] Evaluando prompt engineering...")
        all_results.append(bench_prompt_engineering(TEST_QUERIES))

    if "lora" in args.paths:
        print("\n[3/4] Evaluando LoRA (vLLM)...")
        all_results.append(bench_lora(TEST_QUERIES, port=args.vllm_port))

    if "qlora" in args.paths:
        print("\n[4/4] Evaluando QLoRA (vLLM)...")
        all_results.append(bench_qlora(TEST_QUERIES, port=args.vllm_port))

    # Tabla comparativa
    df = build_comparison_table(all_results)

    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA")
    print("=" * 60)
    print(df.to_string(index=False))

    # Guardar
    save_results(all_results, df)

    # Decisión
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DECISIÓN")
    print("=" * 60)
    best = df.iloc[0]
    print(f"  Mejor calidad:  {best['Ruta']} (score: {best['Calidad (avg)']:.2f})")
    lowest_latency = df.loc[df["Latencia (s)"].idxmin()]
    print(f"  Menor latencia: {lowest_latency['Ruta']} ({lowest_latency['Latencia (s)']:.2f}s)")
    print()
    print("  Recomendación:")
    if best["Ruta"] in ("lora", "qlora"):
        print("    Fine-tuning mejora la calidad. Evalúa si el costo de")
        print("    entrenamiento justifica la mejora vs prompt engineering.")
    else:
        print("    Prompt engineering es suficiente. El fine-tuning no justifica")
        print("    el costo adicional de entrenamiento.")

    print("\n✅ Benchmark completado")


if __name__ == "__main__":
    main()
