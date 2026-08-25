"""Silver layer — stratified train/test split."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from etl_pipeline.utils import get_s3_client, load_config


def run(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the Silver split step.

    Reads the cleaned Parquet, performs an 80/20 stratified split by urgency,
    and writes train/test Parquets to MinIO.
    """
    config = config or load_config()
    client = get_s3_client()

    clean_key: str = config["silver"]["output"]["clean_parquet"]
    train_key: str = config["silver"]["output"]["train_parquet"]
    test_key: str = config["silver"]["output"]["test_parquet"]
    bucket_clean: str = config["minio"]["bucket_clean"]
    test_size: float = config["silver"]["split"]["test_size"]
    random_state: int = config["silver"]["split"]["random_state"]
    stratify_col: str = config["silver"]["split"]["stratify_column"]

    print("[silver:split] Loading cleaned data from MinIO ...")
    obj = client.get_object(Bucket=bucket_clean, Key=clean_key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    print(f"[silver:split] Loaded {len(df):,} rows")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[stratify_col],
    )
    print(f"[silver:split] Train: {len(train_df):,} | Test: {len(test_df):,}")

    for label, frame, key in [("train", train_df, train_key), ("test", test_df, test_key)]:
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False, compression="snappy", engine="pyarrow")
        buf.seek(0)
        client.put_object(
            Bucket=bucket_clean,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        print(f"[silver:split] Saved {label} → s3://{bucket_clean}/{key}")

    return train_df, test_df


if __name__ == "__main__":
    run()
