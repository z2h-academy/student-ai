"""Shared utilities for the ETL pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "config.yml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    config_path = Path(path) if path else _CONFIG_PATH
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_s3_client() -> Any:
    """Create and return a boto3 S3 client connected to MinIO."""
    endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minio-access-key")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minio-secret-key")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def ensure_bucket(client: Any, bucket_name: str) -> None:
    """Create the bucket if it does not exist."""
    existing = client.list_buckets().get("Buckets", [])
    bucket_names = {b["Name"] for b in existing}
    if bucket_name not in bucket_names:
        client.create_bucket(Bucket=bucket_name)
        print(f"[utils] Bucket '{bucket_name}' created.")
    else:
        print(f"[utils] Bucket '{bucket_name}' already exists.")


def ensure_buckets(config: dict[str, Any]) -> None:
    """Ensure all three medallion buckets exist."""
    client = get_s3_client()
    buckets = [
        config["minio"]["bucket_raw"],
        config["minio"]["bucket_clean"],
        config["minio"]["bucket_gold"],
    ]
    for bucket in buckets:
        ensure_bucket(client, bucket)
