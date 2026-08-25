"""Silver layer — clean and enrich the raw reviews dataset."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from etl_pipeline.utils import get_s3_client, load_config


def compute_urgencia(score: int) -> int:
    """Map a 1-5 star score to an urgency level (1=critical … 5=positive)."""
    if score <= 1:
        return 1
    if score == 2:
        return 2
    if score == 3:
        return 3
    if score == 4:
        return 4
    return 5


def run(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute the Silver cleaning step.

    Reads the raw CSV from the Bronze bucket, cleans it, and writes a Parquet
    file to the Silver bucket.
    """
    config = config or load_config()
    client = get_s3_client()

    raw_csv_key: str = config["silver"]["input"]["raw_csv"]
    bucket_raw: str = config["minio"]["bucket_raw"]
    bucket_clean: str = config["minio"]["bucket_clean"]
    output_key: str = config["silver"]["output"]["clean_parquet"]
    text_col: str = config["silver"]["cleaning"]["text_column"]
    summary_col: str = config["silver"]["cleaning"]["summary_column"]
    score_col: str = config["silver"]["cleaning"]["score_column"]
    required_cols: list[str] = config["silver"]["cleaning"]["required_columns"]

    print("[silver:clean] Loading raw CSV from MinIO ...")
    obj = client.get_object(Bucket=bucket_raw, Key=raw_csv_key)
    raw_bytes: bytes = obj["Body"].read()
    df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", dtype=str)
    print(f"[silver:clean] Raw shape: {df.shape}")

    # Drop rows missing critical columns
    df.dropna(subset=required_cols, inplace=True)
    print(f"[silver:clean] After dropping NAs: {df.shape}")

    # Cast Score to numeric
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df.dropna(subset=[score_col], inplace=True)
    df[score_col] = df[score_col].astype(int)

    # Derived columns
    df["review_length"] = df[text_col].str.len().fillna(0).astype(int)
    df["word_count"] = df[text_col].str.split().str.len().fillna(0).astype(int)
    df["urgencia"] = df[score_col].apply(compute_urgencia)

    df.sort_values(by="Id", kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[silver:clean] Final shape: {df.shape}")
    print(f"[silver:clean] Urgencia distribution:\n{df['urgencia'].value_counts().sort_index()}")

    # Write to MinIO as Parquet
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
    buf.seek(0)
    client.put_object(
        Bucket=bucket_clean,
        Key=output_key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    print(f"[silver:clean] Saved to s3://{bucket_clean}/{output_key}")

    return df


if __name__ == "__main__":
    run()
