"""
build_finetune_dataset.py
Construye el dataset de fine-tuning a partir de:
  - state/ de P4 (conversaciones agente↔cliente, tickets resueltos)
  - gold/amazon_reviews/ (reseñas con urgencia clasificada)

Output: gold/finetune/dataset.jsonl con formato (instruction, input, output).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR.parent / "4-detective" / "state"
REVIEWS_DIR = Path(os.getenv(
    "FINETUNE_DATA_PATH",
    str(BASE_DIR.parent / "1-triage" / "gold" / "amazon_reviews"),
))
OUTPUT_DIR = BASE_DIR / "gold" / "finetune"
OUTPUT_FILE = OUTPUT_DIR / "dataset.jsonl"
MIN_OUTPUT_LEN = 20
MAX_INPUT_LEN = 512

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
TRIAGE_TEMPLATE: dict[str, str] = {
    "instruction": (
        "Clasifica la siguiente reseña de Amazon en una categoría de urgencia "
        "(crítica, negativa, neutral, positiva) y justifica brevemente."
    ),
    "template": (
        "Reseña: {text}\n\n"
        "Clasificación de urgencia:"
    ),
}

RESOLVER_TEMPLATE: dict[str, str] = {
    "instruction": (
        "Eres un agente de soporte del Z2H-Shop. Resuelve la consulta del "
        "seller basándote en el contexto de la conversación y la política "
        "de la tienda."
    ),
    "template": (
        "Consulta del seller: {query}\n"
        "Contexto: {context}\n\n"
        "Respuesta del agente:"
    ),
}

ANSWER_REVIEW_TEMPLATE: dict[str, str] = {
    "instruction": (
        "Genera una respuesta profesional del vendedor para esta reseña de "
        "Amazon. Sé empático, resuelve el problema y ofrece alternativas."
    ),
    "template": (
        "Reseña del cliente ({urgency}): {text}\n\n"
        "Respuesta del vendedor:"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """Hash determinístico para deduplicación."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """Limpieza básica de texto: normaliza espacios, elimina artefactos."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    return text


def _validate_chat_format(entry: dict[str, str]) -> bool:
    """Valida que un registro tenga las claves requeridas y valores no vacíos."""
    required_keys = ("instruction", "input", "output")
    if not all(k in entry for k in required_keys):
        return False
    if not all(isinstance(entry[k], str) and len(entry[k].strip()) > 0 for k in required_keys):
        return False
    if len(entry["output"]) < MIN_OUTPUT_LEN:
        return False
    if len(entry["input"]) > MAX_INPUT_LEN:
        return False
    return True


def _load_jsonl_or_parquet(path: Path) -> list[dict[str, Any]]:
    """Carga un archivo .jsonl o .parquet y retorna una lista de dicts."""
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"Formato no soportado: {path.suffix}")


# ---------------------------------------------------------------------------
# Sources: state/ (P4)
# ---------------------------------------------------------------------------

def _load_conversations(state_dir: Path) -> list[dict[str, Any]]:
    """Carga conversaciones guardadas por P4 desde state/."""
    conversations: list[dict[str, Any]] = []
    if not state_dir.exists():
        print(f"[WARN] state/ no encontrado en {state_dir}, usando datos sintéticos.")
        return _synthetic_conversations()

    for ext in ("*.jsonl", "*.json", "*.parquet"):
        for path in sorted(state_dir.glob(ext)):
            try:
                records = _load_jsonl_or_parquet(path)
                conversations.extend(records)
            except Exception as exc:
                print(f"[WARN] Error cargando {path}: {exc}")

    if not conversations:
        print("[WARN] state/ vacío, usando datos sintéticos.")
        return _synthetic_conversations()

    return conversations


def _synthetic_conversations() -> list[dict[str, Any]]:
    """Datos sintéticos de ejemplo cuando no hay state/ de P4."""
    return [
        {
            "query": "Mi pedido llegó dañado y el cliente quiere reembolso",
            "context": "Pedido #1234: funda para laptop, daño visible en empaque",
            "response": (
                "Lamento mucho el inconveniente. Para procesar el reembolso, "
                "por favor envíe fotos del daño a nuestro correo. En 24 horas "
                "le confirmamos la resolución. Alternativamente, podemos enviar "
                "un reemplazo sin costo adicional."
            ),
            "urgency": "crítica",
            "source": "synthetic",
        },
        {
            "query": "¿Cómo configuro los precios de envío internacional?",
            "context": "Seller nuevo, categoría electrónica, envío a EE.UU. y Europa",
            "response": (
                "Para configurar envíos internacionales, ve a Configuración > "
                "Envíos > Zonas. Crea una zona para cada región y asigna las "
                "tarifas según peso/dimensión. Recomendamos usar tarifas "
                "competitivas revisando los precios de la competencia."
            ),
            "urgency": "neutral",
            "source": "synthetic",
        },
        {
            "query": "El ranking de mis productos bajó esta semana",
            "context": "SKU-5678: audífonos bluetooth, categoría electrónica",
            "response": (
                "La caída puede deberse a varios factores: 1) Revisa si hay "
                "reseñas negativas recientes. 2) Verifica que el stock no se "
                "haya agotado. 3) Actualiza las palabras clave del título. "
                "4) Revisa si la competencia bajó precios. Puedo ayudarte a "
                "analizar cada punto."
            ),
            "urgency": "negativa",
            "source": "synthetic",
        },
    ]


# ---------------------------------------------------------------------------
# Sources: reviews (P1 gold)
# ---------------------------------------------------------------------------

def _load_reviews(reviews_dir: Path) -> list[dict[str, Any]]:
    """Carga reseñas desde gold/amazon_reviews/."""
    records: list[dict[str, Any]] = []

    for ext in ("*.jsonl", "*.parquet"):
        for path in sorted(reviews_dir.glob(ext)):
            try:
                records.extend(_load_jsonl_or_parquet(path))
            except Exception as exc:
                print(f"[WARN] Error cargando {path}: {exc}")

    if not records:
        print(f"[WARN] No se encontraron reseñas en {reviews_dir}")
    return records


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_from_conversations(
    conversations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Genera registros de fine-tuning a partir de conversaciones P4."""
    dataset: list[dict[str, str]] = []

    for conv in conversations:
        query = _clean_text(str(conv.get("query", conv.get("question", ""))))
        context = _clean_text(str(conv.get("context", conv.get("history", ""))))
        response = _clean_text(str(conv.get("response", conv.get("answer", conv.get("output", "")))))
        urgency = conv.get("urgency", "neutral")

        if not query or not response:
            continue

        entry: dict[str, str] = {
            "instruction": RESOLVER_TEMPLATE["instruction"],
            "input": RESOLVER_TEMPLATE["template"].format(query=query, context=context),
            "output": response,
        }

        if _validate_chat_format(entry):
            dataset.append(entry)

    return dataset


def _build_answer_reviews(
    reviews: list[dict[str, Any]],
    max_samples: int = 500,
) -> list[dict[str, str]]:
    """Genera registros de fine-tuning para responder reseñas."""
    dataset: list[dict[str, str]] = []
    count = 0

    for review in reviews:
        if count >= max_samples:
            break

        text = _clean_text(str(review.get("Text", review.get("text", ""))))
        urgency = str(review.get("urgency", review.get("sentiment", "neutral")))
        response = _clean_text(str(review.get(
            "response",
            review.get("agent_response", review.get("output", "")),
        )))

        if not text:
            continue

        # Si no hay respuesta pre-generada, usar la reseña como contexto
        if not response or len(response) < MIN_OUTPUT_LEN:
            response = (
                f"Estimado cliente, lamentamos la experiencia. "
                f"Su feedback sobre '{text[:100]}...' es muy valioso. "
                f"Trabajaremos para mejorar. Si necesita ayuda adicional, "
                f"no dude en contactarnos."
            )

        entry: dict[str, str] = {
            "instruction": ANSWER_REVIEW_TEMPLATE["instruction"],
            "input": ANSWER_REVIEW_TEMPLATE["template"].format(
                urgency=urgency, text=text[:MAX_INPUT_LEN],
            ),
            "output": response,
        }

        if _validate_chat_format(entry):
            dataset.append(entry)
            count += 1

    return dataset


def _build_triage_dataset(
    reviews: list[dict[str, Any]],
    max_samples: int = 500,
) -> list[dict[str, str]]:
    """Genera registros de fine-tuning para clasificación de urgencia."""
    dataset: list[dict[str, str]] = []
    count = 0

    for review in reviews:
        if count >= max_samples:
            break

        text = _clean_text(str(review.get("Text", review.get("text", ""))))
        urgency = str(review.get("urgency", review.get("sentiment", "neutral")))

        if not text:
            continue

        output_text = (
            f"Categoría: {urgency}\n"
            f"Justificación: La reseña presenta señales de {urgency} "
            f"basándose en el tono y contenido del texto."
        )

        entry: dict[str, str] = {
            "instruction": TRIAGE_TEMPLATE["instruction"],
            "input": TRIAGE_TEMPLATE["template"].format(text=text[:MAX_INPUT_LEN]),
            "output": output_text,
        }

        if _validate_chat_format(entry):
            dataset.append(entry)
            count += 1

    return dataset


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def _deduplicate(dataset: list[dict[str, str]]) -> list[dict[str, str]]:
    """Elimina duplicados por hash de (instruction + input + output)."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []

    for entry in dataset:
        key = _sha256(f"{entry['instruction']}|{entry['input']}|{entry['output']}")
        if key not in seen:
            seen.add(key)
            unique.append(entry)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Construye y guarda el dataset de fine-tuning."""
    print("=" * 60)
    print("BUILD FINETUNE DATASET")
    print("=" * 60)

    # 1. Cargar fuentes
    print(f"\n[1/5] Cargando conversaciones de P4 desde {STATE_DIR}")
    conversations = _load_conversations(STATE_DIR)
    print(f"  → {len(conversations)} conversaciones cargadas")

    print(f"\n[2/5] Cargando reseñas desde {REVIEWS_DIR}")
    reviews = _load_reviews(REVIEWS_DIR)
    print(f"  → {len(reviews)} reseñas cargadas")

    # 2. Generar registros
    print("\n[3/5] Generando registros de fine-tuning")
    dataset_conv = _build_from_conversations(conversations)
    print(f"  → {len(dataset_conv)} registros de conversaciones (resolver)")

    dataset_reviews = _build_answer_reviews(reviews)
    print(f"  → {len(dataset_reviews)} registros de reseñas (responder)")

    dataset_triage = _build_triage_dataset(reviews)
    print(f"  → {len(dataset_triage)} registros de reseñas (triage)")

    # 3. Unir y deduplicar
    all_records = dataset_conv + dataset_reviews + dataset_triage
    print(f"\n[4/5] Total bruto: {len(all_records)} registros")

    unique_records = _deduplicate(all_records)
    print(f"  → {len(unique_records)} registros únicos tras deduplicación")

    # 4. Guardar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for entry in unique_records:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[5/5] Dataset guardado en {OUTPUT_FILE}")
    print(f"  → {len(unique_records)} registros en formato JSONL")
    print(f"  → Tamaño: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    # 5. Stats
    print("\n--- Estadísticas ---")
    instructions = {}
    for rec in unique_records:
        inst = rec["instruction"][:60]
        instructions[inst] = instructions.get(inst, 0) + 1
    for inst, count in sorted(instructions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:4d} × {inst}...")

    print("\n✅ Dataset listo para fine-tuning")


if __name__ == "__main__":
    main()
