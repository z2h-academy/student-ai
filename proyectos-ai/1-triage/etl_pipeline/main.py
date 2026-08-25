"""Pipeline orchestrator — runs Bronze → Silver → Gold in order."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, Any

from etl_pipeline.utils import ensure_buckets, load_config
from etl_pipeline.bronze import extract
from etl_pipeline.silver import transform_silver, split_data, build_kb
from etl_pipeline.gold import build_embeddings


STEPS: dict[str, tuple[str, Callable[[dict[str, Any]], Any]]] = {
    "bronze": ("Bronze — Extract", extract.run),
    "silver_clean": ("Silver — Clean", transform_silver.run),
    "silver_split": ("Silver — Split", split_data.run),
    "silver_kb": ("Silver — Build KB", build_kb.run),
    "gold": ("Gold — Embeddings", build_embeddings.run),
}

SILVER_SEQUENCE = ["silver_clean", "silver_split", "silver_kb"]


def run_pipeline(step: str, config: dict[str, Any]) -> None:
    """Run the requested pipeline step(s)."""
    ensure_buckets(config)

    if step == "all":
        ordered = ["bronze"] + SILVER_SEQUENCE + ["gold"]
    elif step == "silver":
        ordered = SILVER_SEQUENCE
    elif step in STEPS:
        ordered = [step]
    else:
        print(f"[main] Unknown step: {step}", file=sys.stderr)
        sys.exit(1)

    for step_name in ordered:
        label, func = STEPS[step_name]
        print(f"\n{'=' * 60}")
        print(f"[main] {label}")
        print(f"{'=' * 60}")
        t0 = time.perf_counter()
        func(config)
        elapsed = time.perf_counter() - t0
        print(f"[main] {label} completed in {elapsed:.1f}s")

    print(f"\n[main] Pipeline finished (step={step}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage ETL Pipeline — Medallion Architecture"
    )
    parser.add_argument(
        "--step",
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Pipeline step to execute (default: all)",
    )
    args = parser.parse_args()
    config = load_config()
    run_pipeline(args.step, config)


if __name__ == "__main__":
    main()
