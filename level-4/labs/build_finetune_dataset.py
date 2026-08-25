# level-4/labs/build_finetune_dataset.py
"""
Dataset de fine-tuning: interacciones reales del agente + pares sinteticos de la KB.

Fuentes:
  1. PostgreSQL (tabla agente_memoria de Level 3) -> conversaciones user/assistant
     (filtra escalaciones: no son buenas respuestas para entrenar)
  2. knowledge_base.md -> pares pregunta/respuesta generados con llama3.2

Output: data/finetune_dataset.jsonl (formato chat: {"messages": [...]})

Usage:
    python labs/build_finetune_dataset.py
"""

import json
import os
import re
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "llama3.2"
RUTA_KB = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.md"
RUTA_DATASET = Path(__file__).resolve().parent.parent / "data" / "finetune_dataset.jsonl"

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")

SYSTEM = (
    "Eres 'librarian', un asistente tecnico que responde en espanol "
    "de forma breve y clara, citando fuentes cuando corresponde."
)

FRASES_ESCALACION = ("fuera de mi alcance", "derivada a un humano", "derivado a un humano")


def conversaciones_de_pg() -> list[dict]:
    """Extrae pares user/assistant de la tabla agente_memoria."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sesion, rol, contenido FROM agente_memoria ORDER BY id"
        )
        filas = cur.fetchall()
    conn.close()

    por_sesion: dict[str, list[tuple[str, str]]] = {}
    for sesion, rol, contenido in filas:
        por_sesion.setdefault(sesion, []).append((rol, contenido))

    ejemplos = []
    for mensajes in por_sesion.values():
        pendiente = None
        for rol, contenido in mensajes:
            if rol == "user":
                pendiente = contenido
            elif rol == "assistant" and pendiente:
                es_escalacion = any(f in contenido.lower() for f in FRASES_ESCALACION)
                if not es_escalacion and len(contenido) > 30:
                    ejemplos.append(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": pendiente},
                                {"role": "assistant", "content": contenido},
                            ]
                        }
                    )
                pendiente = None
    return ejemplos


def parrafos_kb() -> list[str]:
    """Lee los parrafos largos de la KB (los que valen como respuesta)."""
    bloques = RUTA_KB.read_text(encoding="utf-8").split("\n\n")
    return [
        b.strip().replace("\n", " ")
        for b in bloques
        if b.strip() and not b.strip().startswith("#") and len(b.strip()) > 120
    ]


def generar_par(parrafo: str) -> dict | None:
    """Genera un par pregunta/respuesta a partir de un parrafo usando llama3.2."""
    prompt = (
        f"A partir del siguiente parrafo tecnico:\n\n{parrafo}\n\n"
        "Genera UNA pregunta de estudiante y su respuesta breve y precisa "
        "(la respuesta debe poder responderse SOLO con el parrafo). "
        'Responde SOLO con JSON: {"pregunta": "...", "respuesta": "..."}'
    )
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat", json=payload, timeout=90
        )
        response.raise_for_status()
        texto = response.json()["message"]["content"]
        match = re.search(r"\{.*\}", texto, re.DOTALL)
        if not match:
            return None
        datos = json.loads(match.group(0))
        if not datos.get("pregunta") or not datos.get("respuesta"):
            return None
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": datos["pregunta"]},
                {"role": "assistant", "content": datos["respuesta"]},
            ]
        }
    except Exception as e:  # noqa: BLE001
        print(f"  (parrafo descartado: {type(e).__name__})")
        return None


def main() -> None:
    print("=== DATASET DE FINE-TUNING ===")

    print("\n1. Interacciones reales desde PostgreSQL...")
    reales = conversaciones_de_pg()
    print(f"   {len(reales)} conversaciones utiles (escalaciones filtradas)")

    print("\n2. Pares sinteticos desde knowledge_base.md...")
    sinteticos = []
    for i, parrafo in enumerate(parrafos_kb()[:6], start=1):
        print(f"   generando par {i}/6...", end=" ", flush=True)
        par = generar_par(parrafo)
        if par:
            sinteticos.append(par)
            print("OK")
        else:
            print("descartado")

    dataset = reales + sinteticos
    RUTA_DATASET.write_text(
        "".join(json.dumps(ej, ensure_ascii=False) + "\n" for ej in dataset),
        encoding="utf-8",
    )

    print(f"\nTotal: {len(dataset)} ejemplos -> {RUTA_DATASET.name}")
    print("\nPrimer ejemplo:")
    print(json.dumps(dataset[0]["messages"][1:], ensure_ascii=False, indent=2)[:300])


if __name__ == "__main__":
    main()