# level-3/labs/agent_final.py
"""
Proyecto integrador: asistente multi-agente con memoria y herramientas.

Reune todo el nivel en un sistema:
  - TRIAGE: clasifica la consulta
  - RESOLVER: usa el RAG (kb_search + responder) con citas
  - ESCALADOR: deriva a humano cuando no puede resolver
  - MEMORIA: persiste cada conversacion en PostgreSQL

Usage:
    python labs/agent_final.py
"""

import json
import os
import sys
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from librarian_pkg.generation import generar
from librarian_pkg.kb import indexar_base
from librarian_pkg.retrieval import recuperar

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")

RUTA_KB = "data/knowledge_base.md"


def _llamar(prompt: str, system: str) -> str:
    """Una llamada de chat a Ollama."""
    payload = {
        "model": MODELO,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ---------- MEMORIA (PostgreSQL) ----------

def _conectar() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    )


def preparar_memoria() -> None:
    """Crea las tablas de memoria si no existen."""
    with _conectar() as conn:
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS escalaciones (
                    id SERIAL PRIMARY KEY,
                    consulta TEXT NOT NULL,
                    motivo TEXT NOT NULL,
                    creado_en TIMESTAMP DEFAULT now()
                )
                """
            )
    print("Memoria lista (tablas agente_memoria y escalaciones).")


def guardar(sesion: str, rol: str, contenido: str) -> None:
    """Persiste un mensaje de la conversacion."""
    with _conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agente_memoria (sesion, rol, contenido) "
                "VALUES (%s, %s, %s)",
                (sesion, rol, contenido),
            )


def history_lookup(sesion: str, limite: int = 4) -> list[tuple[str, str]]:
    """Recupera el historial de una sesion."""
    with _conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rol, contenido FROM agente_memoria "
                "WHERE sesion = %s ORDER BY id DESC LIMIT %s",
                (sesion, limite),
            )
            filas = cur.fetchall()
    return [(rol, contenido) for rol, contenido in reversed(filas)]


def escalar(consulta: str, motivo: str) -> None:
    """Registra una escalacion a humano."""
    with _conectar() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO escalaciones (consulta, motivo) VALUES (%s, %s)",
                (consulta, motivo),
            )
    print(f"  [ESCALADOR] Derivado a humano: {motivo}")


# ---------- AGENTES ----------

def triage(consulta: str) -> str:
    """Clasifica: 'kb', 'otro' o 'urgente'."""
    categoria = _llamar(
        "Clasifica en UNA de: 'kb', 'otro', 'urgente'.\n"
        "Ejemplos:\n"
        "- 'Que es una API REST?' -> kb\n"
        "- 'Como funciona el RAG?' -> kb\n"
        "- 'Hola, como estas?' -> otro\n"
        "- 'Se cayo el sistema, ayuda!' -> urgente\n"
        f"Consulta: {consulta}\n"
        "Responde SOLO con la categoria.",
        "Eres un clasificador estricto.",
    ).lower()
    return categoria if categoria in ("kb", "otro", "urgente") else "otro"


def resolver(consulta: str) -> str:
    """Responde con el RAG y citas de fuente."""
    contexto = recuperar(consulta, n=2)
    return generar(consulta, contexto)


# ---------- FLUJO COMPLETO ----------

def atender(sesion: str, consulta: str) -> str:
    """Flujo completo: memoria -> triage -> resolver/escalar -> guardar."""
    guardar(sesion, "user", consulta)

    print(f"  [TRIAGE] Clasificando...")
    categoria = triage(consulta)
    print(f"  [TRIAGE] -> {categoria}")

    if categoria == "urgente":
        escalar(consulta, "consulta urgente")
        respuesta = "Tu consulta fue derivada a un humano. Te respondera pronto."
    elif categoria == "kb":
        respuesta = resolver(consulta)
    else:
        respuesta = (
            "Eso esta fuera de mi alcance (solo respondo con mi base de "
            "conocimiento). Lo derivo a un humano."
        )
        escalar(consulta, "fuera de alcance del RAG")

    guardar(sesion, "assistant", respuesta)
    return respuesta


def main() -> None:
    print("=== AGENTE FINAL: multi-agente con memoria ===")
    print("Indexando KB...")
    indexar_base(RUTA_KB)
    preparar_memoria()
    print()

    sesion = "sesion-final"
    consultas = [
        "Que es una API REST?",
        "Como funciona el RAG?",
        "Necesito ayuda urgente, se cayo todo!",
    ]

    for consulta in consultas:
        print(f"\nConsulta: {consulta}")
        print(f"Respuesta: {atender(sesion, consulta)[:120]}...")

    print("\n=== HISTORIAL DE LA SESION ===")
    for rol, contenido in history_lookup(sesion):
        print(f"  [{rol}] {contenido[:70]}...")


if __name__ == "__main__":
    main()