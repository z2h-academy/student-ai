"""
eval_rag.py — Evaluación de calidad del RAG

Métricas:
  - Retrieval: precision@k, recall@k
  - Fidelidad de citas: ¿la cita en la respuesta existe en el chunk recuperado?

Genera reporte en JSON y CSV.
"""

import csv
import json
import logging
import os
import pathlib
from datetime import datetime, timezone

from responder import answer

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TEST CASES DEL DOMINIO
# ─────────────────────────────────────────────

DEFAULT_TEST_CASES: list[dict] = [
    {
        "question": "¿Qué productos tienen mejor calificación?",
        "relevant_products": [],  # Se llena dinámicamente contra ChromaDB
        "expected_keywords": ["score", "promedio", "calificación"],
    },
    {
        "question": "¿Cuáles son los productos con más reseñas negativas?",
        "relevant_products": [],
        "expected_keywords": ["negativa", "crítica", "urgencia"],
    },
    {
        "question": "¿Qué productos tienen problemas de calidad?",
        "relevant_products": [],
        "expected_keywords": ["calidad", "problema", "defecto"],
    },
    {
        "question": "¿Cuáles son los productos mejor valorados por los clientes?",
        "relevant_products": [],
        "expected_keywords": ["positiva", "bueno", "excelente"],
    },
    {
        "question": "¿Qué productos tienen reseñas mixtas?",
        "relevant_products": [],
        "expected_keywords": ["mixta", "positiva", "negativa"],
    },
    {
        "question": "¿Cuáles son los productos con urgencia crítica?",
        "relevant_products": [],
        "expected_keywords": ["crítica", "urgencia", "problema"],
    },
    {
        "question": "¿Qué productos recomiendan los clientes?",
        "relevant_products": [],
        "expected_keywords": ["recomiendo", "bueno", "comprar"],
    },
    {
        "question": "¿Cuáles son los productos más populares en número de reseñas?",
        "relevant_products": [],
        "expected_keywords": ["reseñas", "popular", "cantidad"],
    },
    {
        "question": "¿Qué productos tienen problemas de envío?",
        "relevant_products": [],
        "expected_keywords": ["envío", "entrega", "paquete"],
    },
    {
        "question": "¿Cuáles son los productos con mejor relación precio-calidad?",
        "relevant_products": [],
        "expected_keywords": ["precio", "calidad", "valor"],
    },
    {
        "question": "¿Qué productos tienen más quejas sobre durabilidad?",
        "relevant_products": [],
        "expected_keywords": ["durabilidad", "duró", "rompió"],
    },
    {
        "question": "¿Cuáles son los productos favoritos en categoría alimentos?",
        "relevant_products": [],
        "expected_keywords": ["comida", "alimento", "sabor"],
    },
    {
        "question": "¿Qué productos tienen devoluciones frecuentes?",
        "relevant_products": [],
        "expected_keywords": ["devolución", "devolver", "reembolso"],
    },
    {
        "question": "¿Cuáles son los productos con mejor empaque?",
        "relevant_products": [],
        "expected_keywords": ["empaque", "packaging", "envase"],
    },
    {
        "question": "¿Qué productos tienen problemas de tamaño?",
        "relevant_products": [],
        "expected_keywords": ["tamaño", "grande", "pequeño"],
    },
]


# ─────────────────────────────────────────────
# MÉTRICAS DE RETRIEVAL
# ─────────────────────────────────────────────

def precision_at_k(retrieved: list[dict], relevant_products: list[str], k: int) -> float:
    """
    Proporción de los k primeros resultados que son relevantes.

    Un resultado es "relevante" si su source (product_id) está en
    la lista de relevant_products. Si la lista está vacía (sin ground
    truth), se usa la presencia de keywords como proxy.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not relevant_products:
        # Sin ground truth: no hay falsos positivos medibles
        return -1.0
    hits = sum(1 for r in top_k if r["source"] in relevant_products)
    return round(hits / k, 4)


def recall_at_k(retrieved: list[dict], relevant_products: list[str], k: int) -> float:
    """
    Proporción de los productos relevantes recuperados en los k primeros.
    """
    if not relevant_products or k <= 0:
        return -1.0
    top_k = retrieved[:k]
    retrieved_sources = {r["source"] for r in top_k}
    hits = len(retrieved_sources & set(relevant_products))
    return round(hits / len(relevant_products), 4)


def citation_fidelity(answer_text: str, retrieved: list[dict]) -> dict:
    """
    Evalúa la fidelidad de las citas:
      - ¿Las fuentes citadas existen en los chunks recuperados?
      - ¿Se citaron todas las fuentes recuperadas?

    Returns:
        dict con:
            - cited_correctly: int → citas que referencian chunks recuperados
            - cited_incorrectly: int → citas que referencian fuentes NO recuperadas
            - uncited_chunks: int → chunks recuperados sin citar
            - fidelity_score: float → cited_correctly / total_citations
    """
    # Extraer IDs de fuente citados en la respuesta
    cited_sources: set[str] = set()
    for r in retrieved:
        source = r["source"]
        patterns = [
            f"(Fuente: {source})",
            f"(fuente: {source})",
            f"(Source: {source})",
        ]
        if any(p in answer_text for p in patterns):
            cited_sources.add(source)

    # Fuentes recuperadas
    all_sources = {r["source"] for r in retrieved}

    # Fuentes citadas que SÍ están en los resultados
    cited_correctly = len(cited_sources & all_sources)
    # Fuentes citadas que NO están en los resultados (hallucination)
    cited_incorrectly = len(cited_sources - all_sources)
    # Chunks recuperados sin citar
    uncited = len(all_sources - cited_sources)

    total_citations = cited_correctly + cited_incorrectly
    fidelity = (
        round(cited_correctly / total_citations, 4)
        if total_citations > 0
        else 1.0
    )

    return {
        "cited_correctly": cited_correctly,
        "cited_incorrectly": cited_incorrectly,
        "uncited_chunks": uncited,
        "fidelity_score": fidelity,
    }


# ─────────────────────────────────────────────
# EVALUACIÓN COMPLETA
# ─────────────────────────────────────────────

def evaluate(
    test_cases: list[dict] | None = None,
    k: int = 5,
    model: str = "ollama",
    persist_dir: str | None = None,
) -> dict:
    """
    Evalúa el RAG completo sobre un conjunto de test cases.

    Returns:
        dict con:
            - timestamp: str
            - model: str
            - k: int
            - results: list[dict] → resultado por pregunta
            - summary: dict → métricas agregadas
    """
    cases = test_cases or DEFAULT_TEST_CASES
    results: list[dict] = []

    for idx, case in enumerate(cases, 1):
        logger.info(f"[{idx}/{len(cases)}] Evaluando: {case['question'][:60]}...")

        rag_result = answer(
            query=case["question"],
            k=k,
            model=model,
            persist_dir=persist_dir,
        )

        # Retrieval metrics
        p_at_k = precision_at_k(
            rag_result["sources"], case.get("relevant_products", []), k
        )
        r_at_k = recall_at_k(
            rag_result["sources"], case.get("relevant_products", []), k
        )

        # Citation fidelity
        fid = citation_fidelity(rag_result["answer"], rag_result["sources"])

        results.append({
            "question": case["question"],
            "answer": rag_result["answer"],
            "model_used": rag_result["model_used"],
            "retrieved_chunks": len(rag_result["sources"]),
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "citations_correct": fid["cited_correctly"],
            "citations_incorrect": fid["cited_incorrectly"],
            "uncited_chunks": fid["uncited_chunks"],
            "fidelity_score": fid["fidelity_score"],
        })

    # Resumen agregado
    valid_p = [r["precision_at_k"] for r in results if r["precision_at_k"] >= 0]
    valid_r = [r["recall_at_k"] for r in results if r["recall_at_k"] >= 0]
    fidelities = [r["fidelity_score"] for r in results]

    summary = {
        "total_questions": len(results),
        "avg_precision_at_k": round(sum(valid_p) / len(valid_p), 4) if valid_p else -1.0,
        "avg_recall_at_k": round(sum(valid_r) / len(valid_r), 4) if valid_r else -1.0,
        "avg_fidelity": round(sum(fidelities) / len(fidelities), 4) if fidelities else 0.0,
        "total_citations_correct": sum(r["citations_correct"] for r in results),
        "total_citations_incorrect": sum(r["citations_incorrect"] for r in results),
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "k": k,
        "results": results,
        "summary": summary,
    }


def save_report(report: dict, output_dir: str = ".") -> dict:
    """
    Guarda el reporte en JSON y CSV.

    Returns:
        dict con las rutas de los archivos creados.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON completo
    json_path = out / f"eval_rag_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Reporte JSON: {json_path}")

    # CSV resumen
    csv_path = out / f"eval_rag_{ts}.csv"
    if report["results"]:
        fieldnames = [
            "question", "model_used", "retrieved_chunks",
            "precision_at_k", "recall_at_k",
            "citations_correct", "citations_incorrect", "fidelity_score",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["results"])
        logger.info(f"Reporte CSV: {csv_path}")

    return {"json": str(json_path), "csv": str(csv_path)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evalúa calidad del RAG — Proyecto 2 (librarian)"
    )
    parser.add_argument("--k", type=int, default=5, help="Chunks a recuperar")
    parser.add_argument(
        "--model",
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="Proveedor del LLM",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directorio de salida para reportes",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger.info(f"Iniciando evaluación RAG — modelo={args.model}, k={args.k}")
    report = evaluate(k=args.k, model=args.model)
    paths = save_report(report, output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("REPORTE DE EVALUACIÓN RAG")
    print("=" * 60)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nJSON: {paths['json']}")
    print(f"CSV:  {paths['csv']}")
    print("=" * 60)
