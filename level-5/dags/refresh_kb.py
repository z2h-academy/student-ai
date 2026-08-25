# level-5/dags/refresh_kb.py
"""
DAG de re-fresh: reindexa la KB del RAG via la API librarian.

Orquestacion real: el DAG no indexa directamente — llama al endpoint
POST /api/index del servicio librarian-api (la misma API que consumen
los clientes). Asi el pipeline y el servicio comparten una sola fuente.

Schedule: diario a las 06:00 UTC.

Usage:
    Copiar este archivo a ai-platform/dags/ (el contenedor lo monta).
"""

from datetime import datetime

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

API = "http://librarian-api:8000"

default_args = {
    "owner": "ai-engineering",
    "retries": 2,
    "retry_delay": 300,
}


def refrescar_indice() -> int:
    """Reindexa la base de conocimiento via la API."""
    response = requests.post(f"{API}/api/index", timeout=120)
    response.raise_for_status()
    indexados = response.json()["indexados"]
    print(f"KB reindexada: {indexados} parrafos")
    return indexados


def verificar_salud() -> None:
    """Verifica que la API responda antes de indexar."""
    response = requests.get(f"{API}/health", timeout=30)
    response.raise_for_status()
    print(f"Salud de la API: {response.json()}")


with DAG(
    dag_id="refresh_kb",
    description="Re-fresh diario del indice RAG",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * *",
    catchup=False,
    tags=["level-5", "rag"],
) as dag:
    salud = PythonOperator(task_id="verificar_salud", python_callable=verificar_salud)
    refresh = PythonOperator(task_id="refrescar_indice", python_callable=refrescar_indice)

    salud >> refresh