"""
DAG Airflow — eval_agent

Ejecuta evaluación periódica del agente (P4) con un banco de preguntas de test.
Genera un reporte de calidad con métricas de respuesta.

Schedule: semanal (domingos a las 3 AM)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv

load_dotenv()

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://ollama:11434")
PG_HOST = os.getenv("PG_HOST", "postgres-container")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")

# Banco de evaluación: pregunta, respuesta esperada (substring), categoría
EVAL_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "¿Qué es un transformer en machine learning?",
        "expected": "atención",
        "category": "fundamentals",
    },
    {
        "question": "Explica qué hace la capa de self-attention.",
        "expected": "pesos",
        "category": "fundamentals",
    },
    {
        "question": "¿Qué es un embedding?",
        "expected": "vector",
        "category": "embeddings",
    },
    {
        "question": "¿Qué es RAG?",
        "expected": "retrieval",
        "category": "rag",
    },
    {
        "question": "¿Cómo funciona un agente de LangChain?",
        "expected": "herramienta",
        "category": "agents",
    },
    {
        "question": "¿Qué es fine-tuning?",
        "expected": "ajust",
        "category": "optimization",
    },
    {
        "question": "¿Cuál es la diferencia entre batch y streaming?",
        "expected": "tiempo real",
        "category": "data",
    },
    {
        "question": "¿Qué es un DAG en Airflow?",
        "expected": "tarea",
        "category": "orchestration",
    },
    {
        "question": "¿Qué son los guardrails en AI?",
        "expected": "seguridad",
        "category": "safety",
    },
    {
        "question": "Explica la arquitectura medallion.",
        "expected": "bronce",
        "category": "data",
    },
]

default_args = {
    "owner": "ai-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
}


def _query_agent(question: str) -> dict[str, Any]:
    """Consulta el agente con una pregunta y retorna la respuesta."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": "llama3.1:8b",
        "prompt": question,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_ENDPOINT}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "response": data.get("response", ""),
                "eval_duration": data.get("eval_duration", 0),
                "eval_count": data.get("eval_count", 0),
            }
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return {"response": "", "error": str(exc), "eval_duration": 0, "eval_count": 0}


def _check_answer(response: str, expected: str) -> float:
    """Retorna un score (0.0-1.0) de cuánto contiene la respuesta lo esperado."""
    response_lower = response.lower()
    expected_lower = expected.lower()
    if expected_lower in response_lower:
        return 1.0
    return 0.0


def _run_evaluation(**context: object) -> dict[str, Any]:
    """Ejecuta el banco de preguntas y calcula métricas de calidad."""
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(EVAL_QUESTIONS):
        qa_result = _query_agent(item["question"])
        score = _check_answer(qa_result["response"], item["expected"])

        results.append({
            "index": idx,
            "question": item["question"],
            "category": item["category"],
            "expected": item["expected"],
            "response_preview": qa_result["response"][:200],
            "score": score,
            "eval_duration_ns": qa_result.get("eval_duration", 0),
            "error": qa_result.get("error"),
        })

    total = len(results)
    passed = sum(1 for r in results if r["score"] >= 1.0)
    avg_score = sum(r["score"] for r in results) / total if total else 0.0

    category_scores: dict[str, list[float]] = {}
    for r in results:
        cat = r["category"]
        category_scores.setdefault(cat, []).append(r["score"])

    category_averages = {
        cat: sum(scores) / len(scores) for cat, scores in category_scores.items()
    }

    report = {
        "dag": "eval_agent",
        "execution_date": context["ds"],
        "total_questions": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_score": round(avg_score, 4),
        "category_averages": category_averages,
        "detailed_results": results,
    }

    report_path = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "logs",
        f"eval_agent_report_{context['ds']}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def _store_report(**context: object) -> None:
    """Almacena el reporte de evaluación en PostgreSQL."""
    import psycopg2

    ti = context["ti"]
    report = ti.xcom_pull(task_ids="run_evaluation")
    if not report:
        raise ValueError("No se recibió reporte de evaluación desde XCom.")

    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE SCHEMA IF NOT EXISTS evaluation;
            CREATE TABLE IF NOT EXISTS evaluation.agent_reports (
                id SERIAL PRIMARY KEY,
                execution_date DATE NOT NULL,
                total_questions INT NOT NULL,
                passed INT NOT NULL,
                pass_rate FLOAT NOT NULL,
                avg_score FLOAT NOT NULL,
                category_averages JSONB NOT NULL,
                full_report JSONB NOT NULL,
                stored_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("""
            INSERT INTO evaluation.agent_reports
                (execution_date, total_questions, passed, pass_rate, avg_score,
                 category_averages, full_report)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            report["execution_date"],
            report["total_questions"],
            report["passed"],
            report["pass_rate"],
            report["avg_score"],
            json.dumps(report["category_averages"], ensure_ascii=False),
            json.dumps(report, ensure_ascii=False),
        ))

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="eval_agent",
    default_args=default_args,
    description="Evaluación semanal del agente P4 con banco de preguntas de test",
    schedule_interval="0 3 * * 0",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["evaluation", "agent", "p4", "quality"],
) as dag:

    evaluate = PythonOperator(
        task_id="run_evaluation",
        python_callable=_run_evaluation,
        doc="Ejecuta el banco de preguntas contra el agente",
    )

    store = PythonOperator(
        task_id="store_report",
        python_callable=_store_report,
        doc="Almacena el reporte de evaluación en PostgreSQL",
    )

    evaluate >> store
