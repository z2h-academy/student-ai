# level-5/dags/eval_agent.py
"""
DAG de evaluacion periodica: mide la calidad del RAG con preguntas fijas.

Cada corrida hace 3 preguntas al modelo (via Ollama, misma red del
compose) y cuenta cuantas palabras clave esperadas aparecen en cada
respuesta. El resultado queda en los logs de la tarea (visible en la
UI de Airflow) para seguir la calidad del sistema a lo largo del tiempo.

Schedule: cada 6 horas.

Usage:
    Copiar este archivo a ai-platform/dags/ (el contenedor lo monta).
"""

import json
from datetime import datetime

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

OLLAMA = "http://ollama:11434"
MODELO = "llama3.2"

EVALS = [
    {"pregunta": "Que es una API REST?", "claves": ["http", "api"]},
    {"pregunta": "Como funciona el RAG?", "claves": ["contexto", "recupera"]},
    {"pregunta": "Que es ChromaDB?", "claves": ["vector", "embedding"]},
]

default_args = {
    "owner": "ai-engineering",
    "retries": 1,
    "retry_delay": 300,
}


def responder(pregunta: str) -> str:
    """Una respuesta directa del modelo local."""
    payload = {
        "model": MODELO,
        "messages": [{"role": "user", "content": pregunta}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].lower()


def evaluar_calidad() -> str:
    """Ejecuta las evaluaciones y devuelve el reporte JSON."""
    resultados = []
    for ev in EVALS:
        texto = responder(ev["pregunta"])
        aciertos = sum(1 for c in ev["claves"] if c in texto)
        resultados.append(
            {
                "pregunta": ev["pregunta"],
                "claves": f"{aciertos}/{len(ev['claves'])}",
            }
        )
        print(f"  {ev['pregunta']} -> {aciertos}/{len(ev['claves'])} claves")

    total = sum(int(r["claves"].split("/")[0]) for r in resultados)
    maximo = sum(int(r["claves"].split("/")[1]) for r in resultados)
    print(f"CALIDAD GLOBAL: {total}/{maximo}")
    return json.dumps({"total": total, "maximo": maximo, "detalle": resultados})


with DAG(
    dag_id="eval_agent",
    description="Evaluacion periodica de calidad del RAG",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="0 */6 * * *",
    catchup=False,
    tags=["level-5", "evals"],
) as dag:
    evaluar = PythonOperator(task_id="evaluar_calidad", python_callable=evaluar_calidad)
    evaluar