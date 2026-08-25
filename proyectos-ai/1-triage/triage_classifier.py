"""Triage classifier — train a LogisticRegression on review embeddings."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

load_dotenv()

_BASE = Path(__file__).parent


def _load_parquet_minio(config: dict[str, Any], key: str) -> pd.DataFrame:
    """Load a Parquet file from the Silver bucket."""
    import boto3

    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio-access-key")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio-secret-key")
    bucket = config["minio"]["bucket_clean"]

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )
    import io

    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _load_embeddings_from_silver(
    config: dict[str, Any], key: str, text_col: str = "Text"
) -> tuple[np.ndarray, np.ndarray]:
    """Load cleaned parquet and compute embeddings in chunks.

    Returns (X, y) where X is (n_samples, embedding_dim) and y is urgency labels.
    """
    from sentence_transformers import SentenceTransformer

    df = _load_parquet_minio(config, key)

    model_name = config["gold"]["embedding_model"]
    chunk_size = config["classifier"]["embedding_chunk_size"]
    model = SentenceTransformer(model_name)

    texts = df[text_col].tolist()
    all_emb: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        batch = texts[start : start + chunk_size]
        print(f"  Embedding batch {start // chunk_size + 1} ({len(batch)} texts)")
        emb = model.encode(batch, show_progress_bar=False, batch_size=chunk_size)
        all_emb.append(emb)

    X = np.vstack(all_emb)
    y = df["urgencia"].values
    return X, y


def run(config: dict[str, Any] | None = None) -> None:
    """Train the classifier and save model + report."""
    from etl_pipeline.utils import load_config

    config = config or load_config()

    train_key: str = config["gold"]["output"]["train_embeddings"]
    test_key: str = config["gold"]["output"]["test_embeddings"]

    # Try to load pre-computed embeddings from Gold first
    try:
        from etl_pipeline.gold.build_embeddings import _read_parquet_from_minio
        from etl_pipeline.utils import get_s3_client

        client = get_s3_client()
        bucket_gold = config["minio"]["bucket_gold"]

        print("[clf] Loading pre-computed train embeddings from Gold ...")
        train_df = _read_parquet_from_minio(client, bucket_gold, train_key)
        X_train = np.array(train_df["embedding"].tolist())
        y_train = train_df["urgencia"].values

        print("[clf] Loading pre-computed test embeddings from Gold ...")
        test_df = _read_parquet_from_minio(client, bucket_gold, test_key)
        X_test = np.array(test_df["embedding"].tolist())
        y_test = test_df["urgencia"].values
    except Exception:
        print("[clf] Gold embeddings not found, computing from Silver ...")
        X_train, y_train = _load_embeddings_from_silver(config, train_key)
        X_test, y_test = _load_embeddings_from_silver(config, test_key)

    print(f"[clf] Train: {X_train.shape} | Test: {X_test.shape}")

    solver: str = config["classifier"]["solver"]
    max_iter: int = config["classifier"]["max_iter"]

    print(f"[clf] Training LogisticRegression (solver={solver}, max_iter={max_iter}) ...")
    clf = LogisticRegression(solver=solver, max_iter=max_iter, multi_class="auto")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    model_path = _BASE / config["classifier"]["model_path"]
    report_path = _BASE / config["classifier"]["report_path"]
    cm_path = _BASE / config["classifier"]["confusion_matrix_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_path)
    print(f"[clf] Model saved → {model_path}")

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_dict, fh, indent=2, ensure_ascii=False)
    print(f"[clf] Report saved → {report_path}")

    with open(cm_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"confusion_matrix": cm.tolist(), "labels": sorted(set(y_test).tolist())},
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[clf] Confusion matrix saved → {cm_path}")

    print(f"\n[clf] Accuracy: {accuracy:.4f}")
    print(f"[clf] Classification report:\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    run()
