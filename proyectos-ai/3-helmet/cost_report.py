#!/usr/bin/env python3
"""
Calcula costo estimado por request: tokens * precio_modelo + overhead.
Genera un reporte JSON con el desglose.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "llama3.2-local")
COST_PER_1K_TOKENS: float = float(os.getenv("COST_PER_1K_TOKENS", "0.0"))
OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "")


def estimate_cost(
    tokens_input: int,
    tokens_output: int,
    latency_ms: float,
) -> dict[str, float | str]:
    input_cost = (tokens_input / 1000.0) * COST_PER_1K_TOKENS
    output_cost = (tokens_output / 1000.0) * COST_PER_1K_TOKENS
    total_token_cost = input_cost + output_cost
    latency_penalty = latency_ms * 0.00001
    total_cost = total_token_cost + latency_penalty

    return {
        "model": MODEL_NAME,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "cost_input_usd": round(input_cost, 6),
        "cost_output_usd": round(output_cost, 6),
        "latency_ms": round(latency_ms, 2),
        "latency_penalty_usd": round(latency_penalty, 6),
        "total_cost_usd": round(total_cost, 6),
        "currency": "USD",
    }


def generate_report(
    requests_data: list[dict[str, float]],
    output_path: str = "cost_report.json",
) -> str:
    reports: list[dict[str, float | str]] = []
    total = 0.0

    for req in requests_data:
        report = estimate_cost(
            tokens_input=req.get("tokens_input", 0),
            tokens_output=req.get("tokens_output", 0),
            latency_ms=req.get("latency_ms", 0),
        )
        total += report["total_cost_usd"]
        reports.append(report)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "cost_per_1k_tokens": COST_PER_1K_TOKENS,
        "ollama_endpoint": OLLAMA_ENDPOINT or "(local/mock)",
        "total_requests": len(reports),
        "total_estimated_cost_usd": round(total, 6),
        "requests": reports,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    sample_requests = [
        {"tokens_input": 150, "tokens_output": 80, "latency_ms": 450.0},
        {"tokens_input": 300, "tokens_output": 120, "latency_ms": 820.0},
        {"tokens_input": 100, "tokens_output": 50, "latency_ms": 300.0},
    ]

    output = generate_report(sample_requests)
    print(f"Reporte generado: {output}")

    with open(output, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Total requests: {data['total_requests']}")
    print(f"  Total costo estimado: ${data['total_estimated_cost_usd']:.6f} USD")
    print(f"  Modelo: {data['model']}")

    return None


if __name__ == "__main__":
    sys.exit(main() or 0)
