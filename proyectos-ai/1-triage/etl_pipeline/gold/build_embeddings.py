"""Gold layer — generate embeddings for KB chunks and train/test splits."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from etl_pipeline.utils import get_s3_client, load_config


def _read_parquet_from_minio(client: Any, bucket: str, key: str) -> pd.DataFrame:
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _write_parquet_to_minio(df: pd.DataFrame, client: Any, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
    buf.seek(0)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )


def _embed_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode texts in batches to manage memory."""
    all_embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_num = start // batch_size + 1
        print(f"  [gold] Batch {batch_num}/{total_batches} ({len(batch)} texts)")
        embeddings = model.encode(batch, show_progress_bar=False, batch_size=batch_size)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)


def run(config: dict[str, Any] | None = None) -> None:
    """Execute the Gold embedding step."""
    config = config or load_config()
    client = get_s3_client()

    model_name: str = config["gold"]["embedding_model"]
    batch_size: int = config["gold"]["embedding_batch_size"]
    bucket_clean: str = config["minio"]["bucket_clean"]
    bucket_gold: str = config["minio"]["bucket_gold"]

    print(f"[gold] Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # --- KB chunk embeddings ---
    kb_chunks_key: str = config["gold"]["input"]["kb_chunks"]
    kb_emb_key: str = config["gold"]["output"]["kb_embeddings"]

    print("[gold] Loading KB chunks ...")
    kb_df = _read_parquet_from_minio(client, bucket_clean, kb_chunks_key)
    print(f"[gold] Encoding {len(kb_df):,} KB chunks ...")
    kb_embeddings = _embed_in_batches(model, kb_df["text"].tolist(), batch_size)
    kb_emb_df = pd.DataFrame(
        {
            "chunk_id": kb_df["chunk_id"],
            "embedding": [emb.tolist() for emb in kb_embeddings],
        }
    )
    _write_parquet_to_minio(kb_emb_df, client, bucket_gold, kb_emb_key)
    print(f"[gold] KB embeddings saved → s3://{bucket_gold}/{kb_emb_key}")

    # --- Train embeddings ---
    train_key: str = config["gold"]["input"]["train"]
    train_emb_key: str = config["gold"]["output"]["train_embeddings"]

    print("[gold] Loading train split ...")
    train_df = _read_parquet_from_minio(client, bucket_clean, train_key)
    print(f"[gold] Encoding {len(train_df):,} train reviews ...")
    train_embeddings = _embed_in_batches(
        model, train_df["Text"].tolist(), batch_size
    )
    train_emb_df = pd.DataFrame(
        {
            "Id": train_df["Id"].values,
            "urgencia": train_df["urgencia"].values,
            "embedding": [emb.tolist() for emb in train_embeddings],
        }
    )
    _write_parquet_to_minio(train_emb_df, client, bucket_gold, train_emb_key)
    print(f"[gold] Train embeddings saved → s3://{bucket_gold}/{train_emb_key}")

    # --- Test embeddings ---
    test_key: str = config["gold"]["input"]["test"]
    test_emb_key: str = config["gold"]["output"]["test_embeddings"]

    print("[gold] Loading test split ...")
    test_df = _read_parquet_from_minio(client, bucket_clean, test_key)
    print(f"[gold] Encoding {len(test_df):,} test reviews ...")
    test_embeddings = _embed_in_batches(model, test_df["Text"].tolist(), batch_size)
    test_emb_df = pd.DataFrame(
        {
            "Id": test_df["Id"].values,
            "urgencia": test_df["urgencia"].values,
            "embedding": [emb.tolist() for emb in test_embeddings],
        }
    )
    _write_parquet_to_minio(test_emb_df, client, bucket_gold, test_emb_key)
    print(f"[gold] Test embeddings saved → s3://{bucket_gold}/{test_emb_key}")

    print("[gold] All embeddings generated.")


if __name__ == "__main__":
    run()
