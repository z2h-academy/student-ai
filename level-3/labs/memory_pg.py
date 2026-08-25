# level-3/labs/memory_pg.py
"""
Memoria persistente en PostgreSQL: el agente recuerda entre sesiones.

Guarda los mensajes de cada conversacion en la tabla `agente_memoria`
de PostgreSQL (postgres-container) y puede recuperar el historial de
una sesion anterior (history_lookup).

Usage:
    python labs/memory_pg.py
"""

import os
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")


def conectar() -> psycopg2.extensions.connection:
    """Crea la conexion a PostgreSQL."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )


def crear_tabla() -> None:
    """Crea la tabla de memoria si no existe."""
    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS agente_memoria (
                    id SERIAL PRIMARY KEY,
                    sesion TEXT NOT NULL,
                    rol TEXT NOT NULL,
                    contenido TEXT NOT NULL,
                    creado_en TIMESTAMP DEFAULT now()
                )
                """
            )
    print("Tabla agente_memoria lista.")


def guardar_mensaje(sesion: str, rol: str, contenido: str) -> None:
    """Persiste un mensaje de la conversacion."""
    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agente_memoria (sesion, rol, contenido) "
                "VALUES (%s, %s, %s)",
                (sesion, rol, contenido),
            )
    print(f"Guardado [{rol}] en sesion '{sesion}'.")


def history_lookup(sesion: str, limite: int = 10) -> list[tuple[str, str]]:
    """Recupera los ultimos mensajes de una sesion (memoria entre sesiones)."""
    with conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rol, contenido FROM agente_memoria "
                "WHERE sesion = %s ORDER BY id DESC LIMIT %s",
                (sesion, limite),
            )
            filas = cur.fetchall()
    return [(rol, contenido) for rol, contenido in reversed(filas)]


def main() -> None:
    sesion = "sesion-" + datetime.now().strftime("%Y%m%d-%H%M%S")

    print("=== MEMORIA PERSISTENTE (PostgreSQL) ===")
    crear_tabla()

    guardar_mensaje(sesion, "user", "Me llamo Ana y trabajo en IA.")
    guardar_mensaje(sesion, "assistant", "Hola Ana, gusto en conocerte.")

    print()
    print(f"Recuperando historial de '{sesion}':")
    for rol, contenido in history_lookup(sesion):
        print(f"  [{rol}] {contenido}")

    print()
    print("La memoria sobrevive a que el proceso termine: la proxima")
    print("ejecucion puede recuperar esta sesion con history_lookup().")


if __name__ == "__main__":
    main()