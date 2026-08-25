#!/usr/bin/env python3
"""
Benchmark de progresión completa — Mission Control (P6).

Compara latencia, costo y calidad a lo largo de los 6 proyectos
del AI Engineering Roadmap. Genera tabla comparativa en JSON + CSV
y gráfico matplotlib.

Uso:
    python bench_assistant_progression.py [--output-dir ./output]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Estructura de datos del benchmark
# ---------------------------------------------------------------------------


@dataclass
class ProjectMetrics:
    """Métricas de un proyecto individual."""

    project_id: int
    name: str
    description: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tokens_input: int
    total_tokens_output: int
    cost_usd: float
    quality_score: float
    components: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Reporte completo de progresión."""

    generated_at: str
    projects: list[ProjectMetrics]
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Datos de referencia por proyecto (baseline del bootcamp)
# ---------------------------------------------------------------------------

PROJECT_DEFINITIONS: list[dict[str, Any]] = [
    {
        "project_id": 1,
        "name": "medallion-analytics",
        "description": "Pipeline ETL medallion: Bronze → Silver → Gold",
        "avg_latency_ms": 45.0,
        "p95_latency_ms": 89.0,
        "p99_latency_ms": 142.0,
        "total_tokens_input": 0,
        "total_tokens_output": 0,
        "cost_usd": 0.0,
        "quality_score": 0.92,
        "components": ["etl", "sql", "data-quality"],
    },
    {
        "project_id": 2,
        "name": "rag-knowledge-base",
        "description": "Sistema RAG con embeddings y ChromaDB",
        "avg_latency_ms": 320.0,
        "p95_latency_ms": 680.0,
        "p99_latency_ms": 1200.0,
        "total_tokens_input": 450_000,
        "total_tokens_output": 180_000,
        "cost_usd": 2.85,
        "quality_score": 0.85,
        "components": ["embeddings", "vectorstore", "retrieval", "generation"],
    },
    {
        "project_id": 3,
        "name": "api-integration",
        "description": "API FastAPI con Rate Limiting y auth",
        "avg_latency_ms": 120.0,
        "p95_latency_ms": 250.0,
        "p99_latency_ms": 410.0,
        "total_tokens_input": 120_000,
        "total_tokens_output": 85_000,
        "cost_usd": 1.14,
        "quality_score": 0.90,
        "components": ["fastapi", "rate-limiter", "auth", "caching"],
    },
    {
        "project_id": 4,
        "name": "agentic-systems",
        "description": "Sistema multi-agente con LangChain/AutoGen",
        "avg_latency_ms": 850.0,
        "p95_latency_ms": 2100.0,
        "p99_latency_ms": 4500.0,
        "total_tokens_input": 1_200_000,
        "total_tokens_output": 600_000,
        "cost_usd": 9.60,
        "quality_score": 0.78,
        "components": ["langchain", "autogen", "tools", "memory", "planning"],
    },
    {
        "project_id": 5,
        "name": "fine-tuned-model",
        "description": "Fine-tuning LoRA/QLoRA + evaluación",
        "avg_latency_ms": 180.0,
        "p95_latency_ms": 350.0,
        "p99_latency_ms": 520.0,
        "total_tokens_input": 800_000,
        "total_tokens_output": 200_000,
        "cost_usd": 3.40,
        "quality_score": 0.88,
        "components": ["qlora", "lora", "eval", "dataset"],
    },
    {
        "project_id": 6,
        "name": "mission-control",
        "description": "Centro de comando: observabilidad, orquestación, guardrails, k8s",
        "avg_latency_ms": 95.0,
        "p95_latency_ms": 180.0,
        "p99_latency_ms": 290.0,
        "total_tokens_input": 350_000,
        "total_tokens_output": 150_000,
        "cost_usd": 2.33,
        "quality_score": 0.91,
        "components": [
            "prometheus",
            "grafana",
            "airflow",
            "guardrails",
            "kubernetes",
            "benchmark",
        ],
    },
]


def build_report() -> BenchmarkReport:
    """Construye el reporte de benchmark con datos de referencia."""
    projects = [
        ProjectMetrics(
            project_id=p["project_id"],
            name=p["name"],
            description=p["description"],
            avg_latency_ms=p["avg_latency_ms"],
            p95_latency_ms=p["p95_latency_ms"],
            p99_latency_ms=p["p99_latency_ms"],
            total_tokens_input=p["total_tokens_input"],
            total_tokens_output=p["total_tokens_output"],
            cost_usd=p["cost_usd"],
            quality_score=p["quality_score"],
            components=p["components"],
        )
        for p in PROJECT_DEFINITIONS
    ]

    total_cost = sum(p.cost_usd for p in projects)
    total_tokens = sum(p.total_tokens_input + p.total_tokens_output for p in projects)
    avg_quality = sum(p.quality_score for p in projects) / len(projects)
    avg_latency = sum(p.avg_latency_ms for p in projects) / len(projects)

    summary = {
        "total_projects": len(projects),
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "avg_quality_score": round(avg_quality, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "cost_per_quality_point": round(total_cost / avg_quality, 4) if avg_quality else 0,
        "quality_trend": "improving" if projects[-1].quality_score > projects[0].quality_score else "declining",
    }

    return BenchmarkReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        projects=projects,
        summary=summary,
    )


def save_json(report: BenchmarkReport, output_dir: str) -> str:
    """Guarda el reporte como JSON."""
    path = os.path.join(output_dir, "benchmark_progression.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return path


def save_csv(report: BenchmarkReport, output_dir: str) -> str:
    """Guarda una tabla comparativa como CSV."""
    path = os.path.join(output_dir, "benchmark_progression.csv")
    fieldnames = [
        "project_id",
        "name",
        "description",
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "total_tokens_input",
        "total_tokens_output",
        "cost_usd",
        "quality_score",
        "components",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in report.projects:
            row = {
                "project_id": p.project_id,
                "name": p.name,
                "description": p.description,
                "avg_latency_ms": p.avg_latency_ms,
                "p95_latency_ms": p.p95_latency_ms,
                "p99_latency_ms": p.p99_latency_ms,
                "total_tokens_input": p.total_tokens_input,
                "total_tokens_output": p.total_tokens_output,
                "cost_usd": p.cost_usd,
                "quality_score": p.quality_score,
                "components": "|".join(p.components),
            }
            writer.writerow(row)
    return path


def save_chart(report: BenchmarkReport, output_dir: str) -> str:
    """Genera gráfico matplotlib de progresión y lo guarda como PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[bench] matplotlib no disponible — saltando generación de gráfico.", file=sys.stderr)
        return ""

    names = [f"P{p.project_id}" for p in report.projects]
    latencies = [p.avg_latency_ms for p in report.projects]
    costs = [p.cost_usd for p in report.projects]
    qualities = [p.quality_score for p in report.projects]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AI Engineering Roadmap — Progresión de Benchmarks", fontsize=14, fontweight="bold")

    # Panel 1: Latencia
    axes[0].bar(names, latencies, color="#3498db", edgecolor="white")
    axes[0].set_title("Latencia Promedio (ms)")
    axes[0].set_ylabel("ms")
    for i, v in enumerate(latencies):
        axes[0].text(i, v + max(latencies) * 0.02, f"{v:.0f}", ha="center", fontsize=8)

    # Panel 2: Costo
    axes[1].bar(names, costs, color="#e74c3c", edgecolor="white")
    axes[1].set_title("Costo Acumulado (USD)")
    axes[1].set_ylabel("USD")
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("$%.2f"))
    for i, v in enumerate(costs):
        axes[1].text(i, v + max(costs) * 0.02, f"${v:.2f}", ha="center", fontsize=8)

    # Panel 3: Calidad
    axes[2].bar(names, qualities, color="#2ecc71", edgecolor="white")
    axes[2].set_title("Quality Score")
    axes[2].set_ylabel("Score (0-1)")
    axes[2].set_ylim(0, 1.05)
    for i, v in enumerate(qualities):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "benchmark_progression.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def print_table(report: BenchmarkReport) -> None:
    """Imprime una tabla comparativa en la consola."""
    header = (
        f"{'ID':<4} {'Nombre':<25} {'Latencia(ms)':<14} {'Costo(USD)':<12} "
        f"{'Calidad':<10} {'Tokens':<12}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("  AI ENGINEERING — BENCHMARK DE PROGRESIÓN")
    print(separator)
    print(header)
    print(separator)

    for p in report.projects:
        tokens = p.total_tokens_input + p.total_tokens_output
        print(
            f"P{p.project_id:<3} {p.name:<25} {p.avg_latency_ms:<14.1f} "
            f"${p.cost_usd:<11.2f} {p.quality_score:<10.2f} {tokens:<12,}"
        )

    print(separator)
    s = report.summary
    print(
        f"{'TOTAL':<28} {s['avg_latency_ms']:<14.1f} "
        f"${s['total_cost_usd']:<11.2f} {s['avg_quality_score']:<10.4f} {s['total_tokens']:<12,}"
    )
    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de progresión AI Engineering")
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directorio de salida para los archivos generados",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    report = build_report()

    json_path = save_json(report, output_dir)
    print(f"[bench] JSON: {json_path}")

    csv_path = save_csv(report, output_dir)
    print(f"[bench] CSV:  {csv_path}")

    chart_path = save_chart(report, output_dir)
    if chart_path:
        print(f"[bench] PNG:  {chart_path}")

    print_table(report)


if __name__ == "__main__":
    main()
