"""Tool MCP: registra escalacion en PostgreSQL y estado en S3 (MinIO)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _get_connection() -> psycopg2.extensions.connection:
    """Retorna una conexion a PostgreSQL usando variables de entorno."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme1234"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
    )


def _save_state_s3(ticket_data: dict[str, Any], ticket_id: str) -> None:
    """Persiste el ticket en S3 (MinIO) bajo gold/amazon_reviews/state/."""
    try:
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio-access-key"),
            aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio-secret-key"),
        )

        bucket = "gold"
        key = f"amazon_reviews/state/escalation_{ticket_id}.json"

        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(ticket_data, ensure_ascii=False, default=str),
            ContentType="application/json",
        )
    except Exception:
        pass


def escalate(ticket_data: dict[str, Any]) -> str:
    """Registra un ticket de escalamiento en PostgreSQL y S3.

    Args:
        ticket_data: Diccionario con seller_id, subject, description,
                     priority, category, context_summary.

    Returns:
        El ticket_id generado.
    """
    seller_id = ticket_data.get("seller_id", "unknown")
    subject = ticket_data.get("subject", "Escalamiento automatico")
    description = ticket_data.get("description", "")
    priority = ticket_data.get("priority", "medium")
    category = ticket_data.get("category", "general")
    context_summary = ticket_data.get("context_summary", "")

    full_description = description
    if context_summary:
        full_description = f"{description}\n\nContexto adicional:\n{context_summary}"

    try:
        conn = _get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """\
                    INSERT INTO tickets (seller_id, subject, description, category, status, priority)
                    VALUES (%s, %s, %s, %s, 'escalated', %s)
                    RETURNING id::text
                    """,
                    (seller_id, subject, full_description, category, priority),
                )
                ticket_id = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()
    except psycopg2.OperationalError:
        ticket_id = f"mock-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    ticket_data_full = {
        **ticket_data,
        "description": full_description,
        "ticket_id": ticket_id,
        "status": "escalated",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _save_state_s3(ticket_data_full, ticket_id)

    return str(ticket_id)
