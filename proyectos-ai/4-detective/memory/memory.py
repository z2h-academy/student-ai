"""Memoria de conversacion persistente en PostgreSQL."""

from __future__ import annotations

import os
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    seller_id VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

_CREATE_INDEX_SQL = """\
CREATE INDEX IF NOT EXISTS idx_conversations_seller
ON conversations (seller_id, created_at DESC);
"""


class ConversationMemory:
    """Guarda y recupera el historial de conversaciones por seller."""

    def __init__(self) -> None:
        self._init_table()

    def _get_connection(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "changeme1234"),
            dbname=os.getenv("POSTGRES_DB", "postgres"),
        )

    def _init_table(self) -> None:
        """Crea la tabla conversations si no existe."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                    cur.execute(_CREATE_INDEX_SQL)
                conn.commit()
            finally:
                conn.close()
        except psycopg2.OperationalError:
            pass

    def save_message(self, seller_id: str, role: str, content: str) -> int | None:
        """Guarda un mensaje en la conversacion.

        Args:
            seller_id: Identificador del seller.
            role: 'user' o 'assistant'.
            content: Contenido del mensaje.

        Returns:
            El id del registro insertado, o None si falla.
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """\
                        INSERT INTO conversations (seller_id, role, content)
                        VALUES (%s, %s, %s)
                        RETURNING id
                        """,
                        (seller_id, role, content),
                    )
                    row_id = cur.fetchone()[0]
                conn.commit()
                return row_id
            finally:
                conn.close()
        except psycopg2.OperationalError:
            return None

    def get_conversation(
        self, seller_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recupera la conversacion mas reciente de un seller.

        Args:
            seller_id: Identificador del seller.
            limit: Maximo de mensajes a retornar.

        Returns:
            Lista de diccionarios con role, content y created_at.
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """\
                        SELECT role, content, created_at
                        FROM conversations
                        WHERE seller_id = %s
                        ORDER BY created_at ASC
                        LIMIT %s
                        """,
                        (seller_id, limit),
                    )
                    return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()
        except psycopg2.OperationalError:
            return []

    def clear_conversation(self, seller_id: str) -> int:
        """Elimina todos los mensajes de un seller.

        Args:
            seller_id: Identificador del seller.

        Returns:
            Cantidad de registros eliminados.
        """
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM conversations WHERE seller_id = %s",
                        (seller_id,),
                    )
                    deleted = cur.rowcount
                conn.commit()
                return deleted
            finally:
                conn.close()
        except psycopg2.OperationalError:
            return 0

    def get_sellers(self) -> list[str]:
        """Retorna la lista de seller_ids con conversaciones."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT seller_id FROM conversations ORDER BY seller_id"
                    )
                    return [row[0] for row in cur.fetchall()]
            finally:
                conn.close()
        except psycopg2.OperationalError:
            return []
