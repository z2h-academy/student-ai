"""Estado del agente en S3 (MinIO): gold/amazon_reviews/state/."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_STATE_PREFIX = "gold/amazon_reviews/state/"


def _get_s3_client() -> Any:
    """Retorna un cliente S3 (MinIO)."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio-access-key"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio-secret-key"),
    )


def save_state(state_key: str, data: dict[str, Any]) -> str:
    """Serializa y guarda el estado del agente en S3.

    Args:
        state_key: Clave identificadora del estado (sin extension).
        data: Diccionario con el estado a persistir.

    Returns:
        La clave S3 completa donde se guardo.
    """
    s3 = _get_s3_client()
    bucket = "gold"
    key = f"{_STATE_PREFIX}{state_key}.json"

    payload = {
        "state_key": state_key,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, default=str),
        ContentType="application/json",
    )

    return key


def load_state(state_key: str) -> dict[str, Any] | None:
    """Carga el estado del agente desde S3.

    Args:
        state_key: Clave identificadora del estado.

    Returns:
        Diccionario con el estado, o None si no existe.
    """
    try:
        s3 = _get_s3_client()
        bucket = "gold"
        key = f"{_STATE_PREFIX}{state_key}.json"

        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception:
        return None


def list_states() -> list[dict[str, Any]]:
    """Lista todos los estados guardados en S3.

    Returns:
        Lista de diccionarios con metadata de cada estado.
    """
    try:
        s3 = _get_s3_client()
        bucket = "gold"
        prefix = _STATE_PREFIX

        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = response.get("Contents", [])

        states: list[dict[str, Any]] = []
        for obj in contents:
            key = obj["Key"]
            if not key.endswith(".json"):
                continue

            try:
                s3_response = s3.get_object(Bucket=bucket, Key=key)
                content = s3_response["Body"].read().decode("utf-8")
                payload = json.loads(content)
                states.append({
                    "key": key,
                    "state_key": payload.get("state_key", ""),
                    "saved_at": payload.get("saved_at", ""),
                    "size_bytes": obj.get("Size", 0),
                })
            except Exception:
                states.append({
                    "key": key,
                    "state_key": "",
                    "saved_at": "",
                    "size_bytes": obj.get("Size", 0),
                })

        return sorted(states, key=lambda x: x.get("saved_at", ""), reverse=True)
    except Exception:
        return []


def delete_state(state_key: str) -> bool:
    """Elimina un estado de S3.

    Args:
        state_key: Clave identificadora del estado.

    Returns:
        True si se elimino correctamente.
    """
    try:
        s3 = _get_s3_client()
        bucket = "gold"
        key = f"{_STATE_PREFIX}{state_key}.json"

        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False
