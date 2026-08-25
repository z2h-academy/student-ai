"""Bronze layer — extract raw data from source into MinIO."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from etl_pipeline.utils import ensure_bucket, get_s3_client, load_config


def download_file(url: str, dest: Path, retries: int = 3) -> Path:
    """Download a file from *url* to *dest* with retries and progress bar."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading (attempt {attempt})",
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
                    bar.update(len(chunk))
            return dest
        except requests.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {exc}"
                ) from exc
    raise RuntimeError("Unreachable: download retry loop exited unexpectedly")


def run(config: dict[str, Any] | None = None) -> None:
    """Execute the Bronze extraction step."""
    config = config or load_config()
    source_url: str = config["bronze"]["source_url"]
    raw_path: str = config["bronze"]["raw_path"]
    bucket: str = config["minio"]["bucket_raw"]

    tmp_dir = Path(config["bronze"].get("tmp_dir", tempfile.gettempdir())) / "1-triage"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_csv = tmp_dir / "Reviews.csv"

    print("[bronze] Downloading Reviews.csv ...")
    download_file(source_url, local_csv)
    print(f"[bronze] Downloaded to {local_csv} ({local_csv.stat().st_size:,} bytes)")

    client = get_s3_client()
    ensure_bucket(client, bucket)

    print(f"[bronze] Uploading to MinIO → s3://{bucket}/{raw_path}")
    client.upload_file(str(local_csv), bucket, raw_path)
    print("[bronze] Upload complete.")


if __name__ == "__main__":
    run()
