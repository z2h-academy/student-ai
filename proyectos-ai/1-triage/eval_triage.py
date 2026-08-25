"""Evaluation script — generate metrics from the trained triage classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

load_dotenv()

_BASE = Path(__file__).parent


def run(config: dict[str, Any] | None = None) -> None:
    """Load saved model and test embeddings, produce evaluation JSON."""
    from etl_pipeline.utils import load_config

    config = config or load_config()

    model_path = _BASE / config["classifier"]["model_path"]
    report_path = _BASE / config["classifier"]["report_path"]
    cm_path = _BASE / config["classifier"]["confusion_matrix_path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run triage_classifier.py first."
        )

    clf = joblib.load(model_path)
    print(f"[eval] Loaded model from {model_path}")

    # Load test embeddings from Gold
    from etl_pipeline.gold.build_embeddings import _read_parquet_from_minio
    from etl_pipeline.utils import get_s3_client

    client = get_s3_client()
    bucket_gold = config["minio"]["bucket_gold"]
    test_key: str = config["gold"]["output"]["test_embeddings"]

    test_df = _read_parquet_from_minio(client, bucket_gold, test_key)
    X_test = np.array(test_df["embedding"].tolist())
    y_test = test_df["urgencia"].values
    print(f"[eval] Loaded {len(y_test):,} test samples")

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, ensure_ascii=False)
    print(f"[eval] Report saved → {report_path}")

    with open(cm_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"confusion_matrix": cm.tolist(), "labels": sorted(set(y_test).tolist())},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[eval] Confusion matrix saved → {cm_path}")

    print(f"\n[eval] Accuracy: {accuracy:.4f}")
    print(f"[eval] Classification report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    run()
