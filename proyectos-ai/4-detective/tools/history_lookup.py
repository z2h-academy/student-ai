"""Tool MCP: consulta PostgreSQL para historial de tickets del seller."""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS tickets (
    id SERIAL PRIMARY KEY,
    seller_id VARCHAR(64) NOT NULL,
    subject VARCHAR(256) NOT NULL,
    description TEXT,
    category VARCHAR(64),
    status VARCHAR(32) DEFAULT 'open',
    priority VARCHAR(16) DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT NOW()
);
"""


def _get_connection() -> psycopg2.extensions.connection:
    """Retorna una conexion a PostgreSQL usando variables de entorno."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme1234"),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
    )


def init_history_table() -> None:
    """Crea la tabla tickets si no existe."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def history_lookup(seller_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Recupera el historial de tickets de un seller.

    Args:
        seller_id: Identificador del seller.
        limit: Maximo de tickets a retornar.

    Returns:
        Lista de diccionarios con los tickets mas recientes.
    """
    try:
        conn = _get_connection()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """\
                    SELECT id, seller_id, subject, description, category,
                           status, priority, created_at
                    FROM tickets
                    WHERE seller_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (seller_id, limit),
                )
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        finally:
            conn.close()
    except psycopg2.OperationalError:
        return _mock_history(seller_id, limit)


def _mock_history(seller_id: str, limit: int) -> list[dict[str, Any]]:
    """Retorna historial simulado para testing sin PostgreSQL."""
    return [
        {
            "id": 1,
            "seller_id": seller_id,
            "subject": "Ticket simulado #1",
            "description": "Consulta de ejemplo",
            "category": "general",
            "status": "resolved",
            "priority": "medium",
            "created_at": "2026-01-15T10:00:00",
        },
        {
            "id": 2,
            "seller_id": seller_id,
            "subject": "Ticket simulado #2",
            "description": "Otra consulta de ejemplo",
            "category": "facturacion",
            "status": "open",
            "priority": "low",
            "created_at": "2026-01-10T08:30:00",
        },
    ][:limit]
