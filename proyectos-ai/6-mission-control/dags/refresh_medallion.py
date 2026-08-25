"""
DAG Airflow — refresh_medallion

Ejecuta el pipeline ETL de P1 (medallion architecture):
  Bronze → Silver → Gold

Schedule: diario (0 2 * * *)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv

load_dotenv()

PG_HOST = os.getenv("PG_HOST", "postgres-container")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")

default_args = {
    "owner": "ai-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------


def _bronze_extract(**context: object) -> dict[str, int]:
    """Extrae datos crudos y los almacena en la capa bronze."""
    import psycopg2

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
            CREATE SCHEMA IF NOT EXISTS bronze;
            CREATE TABLE IF NOT EXISTS bronze.raw_documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                ingested_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("SELECT COUNT(*) FROM bronze.raw_documents")
        count = cur.fetchone()[0]

        conn.commit()
        return {"bronze_rows": count}
    finally:
        conn.close()


def _silver_transform(**context: object) -> dict[str, int]:
    """Limpia, deduplica y normaliza datos de bronze → silver."""
    import psycopg2

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
            CREATE SCHEMA IF NOT EXISTS silver;
            CREATE TABLE IF NOT EXISTS silver.clean_documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                content_length INT NOT NULL,
                cleaned_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(source, content)
            );
        """)

        cur.execute("""
            INSERT INTO silver.clean_documents (source, content, content_length)
            SELECT DISTINCT ON (source, content)
                source,
                TRIM(content) AS content,
                LENGTH(TRIM(content)) AS content_length
            FROM bronze.raw_documents
            ON CONFLICT (source, content) DO NOTHING;
        """)

        conn.commit()

        cur.execute("SELECT COUNT(*) FROM silver.clean_documents")
        count = cur.fetchone()[0]

        return {"silver_rows": count}
    finally:
        conn.close()


def _gold_load(**context: object) -> dict[str, int]:
    """Genera la capa gold con métricas y agregaciones."""
    import psycopg2

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
            CREATE SCHEMA IF NOT EXISTS gold;
            CREATE TABLE IF NOT EXISTS gold.document_stats (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                total_documents INT NOT NULL,
                avg_content_length FLOAT NOT NULL,
                computed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("""
            INSERT INTO gold.document_stats (source, total_documents, avg_content_length)
            SELECT
                source,
                COUNT(*) AS total_documents,
                AVG(content_length)::FLOAT AS avg_content_length
            FROM silver.clean_documents
            GROUP BY source;
        """)

        conn.commit()

        cur.execute("SELECT COUNT(*) FROM gold.document_stats")
        count = cur.fetchone()[0]

        return {"gold_rows": count}
    finally:
        conn.close()


def _report_metrics(**context: object) -> str:
    """Genera un reporte JSON con las métricas del pipeline."""
    import psycopg2

    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM bronze.raw_documents")
        bronze_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM silver.clean_documents")
        silver_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM gold.document_stats")
        gold_count = cur.fetchone()[0]

        report = {
            "pipeline": "medallion_refresh",
            "execution_date": context["ds"],
            "bronze_rows": bronze_count,
            "silver_rows": silver_count,
            "gold_aggregations": gold_count,
        }

        report_path = os.path.join(
            os.getenv("AIRFLOW_HOME", "/opt/airflow"),
            "logs",
            f"medallion_report_{context['ds']}.json",
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return json.dumps(report, ensure_ascii=False)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="refresh_medallion",
    default_args=default_args,
    description="Pipeline ETL medallion: Bronze → Silver → Gold (diario)",
    schedule_interval="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["medallion", "etl", "p1"],
) as dag:

    bronze = PythonOperator(
        task_id="bronze_extract",
        python_callable=_bronze_extract,
        doc="Extrae datos crudos a la capa bronze",
    )

    silver = PythonOperator(
        task_id="silver_transform",
        python_callable=_silver_transform,
        doc="Limpia y deduplica datos de bronze a silver",
    )

    gold = PythonOperator(
        task_id="gold_load",
        python_callable=_gold_load,
        doc="Genera agregaciones en la capa gold",
    )

    report = PythonOperator(
        task_id="report_metrics",
        python_callable=_report_metrics,
        doc="Genera reporte JSON de métricas del pipeline",
    )

    bronze >> silver >> gold >> report
