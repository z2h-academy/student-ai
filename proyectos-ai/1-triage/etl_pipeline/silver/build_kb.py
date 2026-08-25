"""Silver layer — build a knowledge base by chunking reviews into sentences."""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

from etl_pipeline.utils import get_s3_client, load_config

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_sentences: int = 5) -> list[str]:
    """Split *text* into sentence-level chunks of up to *max_sentences* each."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = _SENTENCE_RE.split(text.strip())
    chunks: list[str] = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i : i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute the Silver KB build step."""
    config = config or load_config()
    client = get_s3_client()

    clean_key: str = config["silver"]["output"]["clean_parquet"]
    kb_key: str = config["silver"]["output"]["chunks_parquet"]
    bucket_clean: str = config["minio"]["bucket_clean"]
    max_sent: int = config["silver"]["kb"]["chunk_max_sentences"]
    text_col: str = config["silver"]["cleaning"]["text_column"]

    print("[silver:kb] Loading cleaned data ...")
    obj = client.get_object(Bucket=bucket_clean, Key=clean_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"[silver:kb] Loaded {len(df):,} reviews")

    records: list[dict[str, Any]] = []
    chunk_id = 0
    for _, row in df.iterrows():  # noqa: B007 — index not needed
        text = row[text_col]
        source_id = str(row["Id"])
        chunks = chunk_text(text, max_sentences=max_sent)
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"chunk_{chunk_id:07d}",
                    "text": chunk,
                    "source": source_id,
                    "chunk_index": idx,
                }
            )
            chunk_id += 1

    kb_df = pd.DataFrame.from_records(records)
    print(f"[silver:kb] Created {len(kb_df):,} chunks from {len(df):,} reviews")

    buf = io.BytesIO()
    kb_df.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
    buf.seek(0)
    client.put_object(
        Bucket=bucket_clean,
        Key=kb_key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"[silver:kb] Saved to s3://{bucket_clean}/{kb_key}")

    return kb_df


if __name__ == "__main__":
    run()
