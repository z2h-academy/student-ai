# level-3/labs/detective.py
"""
Patrones de produccion: triage / resolver / escalador.

El flujo del 'detective':
  1. TRIAGE clasifica la consulta (kb / otro / urgente)
  2. RESOLVER intenta responder con las tools del RAG (kb_search, responder)
  3. ESCALADOR registra en PostgreSQL las consultas que no se resuelven

Usage:
    python labs/detective.py
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


# ---------- AGENTE: TRIAGE ----------

def triage(consulta: str) -> str:
    """Clasifica la consulta: 'kb', 'otro' o 'urgente'."""
    prompt = (
        "Clasifica la siguiente consulta en UNA de estas categorias:\n"
        "- 'kb': pregunta de conocimiento que podria estar en una base de datos\n"
        "- 'otro': saludo, opinion o tema general\n"
        "- 'urgente': problema critico, error, panico o emergencia\n"
        "Ejemplos:\n"
        "- 'Que es una API REST?' -> kb\n"
        "- 'Como funciona el RAG?' -> kb\n"
        "- 'Hola, como estas?' -> otro\n"
        "- 'Se cayo el sistema, ayuda!' -> urgente\n"
        f"Consulta: {consulta}\n"
        "Responde SOLO con la categoria, sin explicaciones."
    )
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    categoria = response.json()["message"]["content"].strip().lower()

    if categoria not in ("kb", "otro", "urgente"):
        categoria = "otro"
    return categoria


# ---------- AGENTE: RESOLVER ----------

def resolver(consulta: str) -> str:
    """Responde con el RAG: recupera contexto y genera respuesta con citas."""
    contexto = recuperar(consulta, n=2)
    return generar(consulta, contexto)


# ---------- AGENTE: ESCALADOR ----------

def crear_tabla_escalaciones() -> None:
    """Crea la tabla de escalaciones en PostgreSQL."""
    with psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    ) as conn:
        with conn.cursor() as cur:
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
    print("Tabla escalaciones lista.")


def escalate_human(consulta: str, motivo: str) -> None:
    """Registra una escalacion para un humano."""
    with psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO escalaciones (consulta, motivo) VALUES (%s, %s)",
                (consulta, motivo),
            )
    print(f"ESCALADO a humano: '{consulta}' ({motivo})")


# ---------- FLUJO COMPLETO ----------

def detective(consulta: str) -> None:
    """Flujo completo: triage -> resolver o escalar."""
    print(f"\nConsulta: {consulta}")

    categoria = triage(consulta)
    print(f"TRIAGE -> {categoria}")

    if categoria == "kb":
        print("RESOLVER -> intentando con el RAG...")
        print(f"Respuesta: {resolver(consulta)[:130]}...")
    elif categoria == "urgente":
        escalate_human(consulta, "consulta urgente")
    else:
        escalate_human(consulta, "fuera de alcance del RAG")


def main() -> None:
    print("=== DETECTIVE: TRIAGE / RESOLVER / ESCALADOR ===")
    print("Indexando KB...")
    indexar_base(RUTA_KB)
    crear_tabla_escalaciones()

    detective("Que es una API REST?")
    detective("Necesito ayuda urgente, el sistema se cayo!")
    detective("Hola, como estas?")


if __name__ == "__main__":
    main()