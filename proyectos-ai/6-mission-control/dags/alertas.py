"""
DAG Airflow — alertas

Monitorea las métricas de observabilidad (P6) y genera alertas
si la calidad de algún componente cae por debajo de un umbral.

Schedule: cada 6 horas
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from dotenv import load_dotenv

load_dotenv()

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
PG_HOST = os.getenv("PG_HOST", "postgres-container")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme1234")
PG_DB = os.getenv("PG_DB", "postgres")

# Umbrales de alerta
QUALITY_THRESHOLD: float = float(os.getenv("ALERT_QUALITY_THRESHOLD", "0.7"))
DRIFT_THRESHOLD: float = float(os.getenv("ALERT_DRIFT_THRESHOLD", "0.3"))
ERROR_RATE_THRESHOLD: float = float(os.getenv("ALERT_ERROR_RATE_THRESHOLD", "0.05"))
LATENCY_THRESHOLD: float = float(os.getenv("ALERT_LATENCY_THRESHOLD", "5.0"))

default_args = {
    "owner": "ai-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
}


def _prom_query(query: str) -> dict[str, object]:
    """Ejecuta una query instantánea contra Prometheus."""
    import urllib.parse

    params = urllib.parse.urlencode({"query": query})
    url = f"{PROMETHEUS_URL}/api/v1/query?{params}"

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data  # type: ignore[no-any-return]
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return {"status": "error", "error": str(exc), "data": {"result": []}}


def _check_quality(**context: object) -> list[dict[str, str]]:
    """Verifica que la calidad de cada componente esté por encima del umbral."""
    alerts: list[dict[str, str]] = []
    query = 'mission_control_quality_score'

    result = _prom_query(query)
    for item in result.get("data", {}).get("result", []):
        component = item.get("metric", {}).get("component", "unknown")
        value = float(item.get("value", [0, 0])[1])
        if value < QUALITY_THRESHOLD:
            alerts.append({
                "type": "quality_low",
                "severity": "warning",
                "component": component,
                "message": f"Calidad de {component}: {value:.4f} < {QUALITY_THRESHOLD}",
            })

    return alerts


def _check_drift(**context: object) -> list[dict[str, str]]:
    """Verifica que el drift de calidad no exceda el umbral."""
    alerts: list[dict[str, str]] = []
    query = "mission_control_drift_score"

    result = _prom_query(query)
    for item in result.get("data", {}).get("result", []):
        component = item.get("metric", {}).get("component", "unknown")
        value = float(item.get("value", [0, 0])[1])
        if value > DRIFT_THRESHOLD:
            alerts.append({
                "type": "drift_high",
                "severity": "warning",
                "component": component,
                "message": f"Drift de {component}: {value:.4f} > {DRIFT_THRESHOLD}",
            })

    return alerts


def _check_error_rate(**context: object) -> list[dict[str, str]]:
    """Verifica la tasa de errores global."""
    alerts: list[dict[str, str]] = []
    query = (
        'rate(mission_control_requests_total{status="500"}[5m]) / '
        'rate(mission_control_requests_total[5m])'
    )

    result = _prom_query(query)
    for item in result.get("data", {}).get("result", []):
        endpoint = item.get("metric", {}).get("endpoint", "unknown")
        value = float(item.get("value", [0, 0])[1])
        if value > ERROR_RATE_THRESHOLD:
            alerts.append({
                "type": "error_rate_high",
                "severity": "critical",
                "component": endpoint,
                "message": f"Error rate en {endpoint}: {value:.4f} > {ERROR_RATE_THRESHOLD}",
            })

    return alerts


def _check_latency(**context: object) -> list[dict[str, str]]:
    """Verifica la latencia P99 de los endpoints."""
    alerts: list[dict[str, str]] = []
    query = 'histogram_quantile(0.99, rate(mission_control_request_latency_seconds_bucket[5m]))'

    result = _prom_query(query)
    for item in result.get("data", {}).get("result", []):
        endpoint = item.get("metric", {}).get("endpoint", "unknown")
        value = float(item.get("value", [0, 0])[1])
        if value > LATENCY_THRESHOLD:
            alerts.append({
                "type": "latency_high",
                "severity": "warning",
                "component": endpoint,
                "message": f"Latencia P99 en {endpoint}: {value:.2f}s > {LATENCY_THRESHOLD}s",
            })

    return alerts


def _collect_alerts(**context: object) -> list[dict[str, str]]:
    """Recolecta todas las alertas de las 4 verificaciones."""
    all_alerts: list[dict[str, str]] = []
    all_alerts.extend(_check_quality(**context))
    all_alerts.extend(_check_drift(**context))
    all_alerts.extend(_check_error_rate(**context))
    all_alerts.extend(_check_latency(**context))

    return all_alerts


def _store_alerts(**context: object) -> dict[str, object]:
    """Almacena las alertas generadas y retorna un resumen."""
    import psycopg2

    ti = context["ti"]
    alerts = ti.xcom_pull(task_ids="collect_alerts") or []

    summary = {
        "dag": "alertas",
        "execution_date": context["ds"],
        "total_alerts": len(alerts),
        "critical": sum(1 for a in alerts if a.get("severity") == "critical"),
        "warning": sum(1 for a in alerts if a.get("severity") == "warning"),
        "alerts": alerts,
    }

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
            CREATE SCHEMA IF NOT EXISTS monitoring;
            CREATE TABLE IF NOT EXISTS monitoring.alerts (
                id SERIAL PRIMARY KEY,
                execution_date DATE NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        for alert in alerts:
            cur.execute("""
                INSERT INTO monitoring.alerts (execution_date, alert_type, severity, component, message)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                context["ds"],
                alert.get("type", "unknown"),
                alert.get("severity", "warning"),
                alert.get("component", "unknown"),
                alert.get("message", ""),
            ))

        conn.commit()
    finally:
        conn.close()

    report_path = os.path.join(
        os.getenv("AIRFLOW_HOME", "/opt/airflow"),
        "logs",
        f"alertas_report_{context['ds']}.json",
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="alertas",
    default_args=default_args,
    description="Monitoreo de métricas y alertas cada 6 horas",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["monitoring", "alerts", "observability", "p6"],
) as dag:

    collect = PythonOperator(
        task_id="collect_alerts",
        python_callable=_collect_alerts,
        doc="Recolecta alertas de Prometheus",
    )

    store = PythonOperator(
        task_id="store_alerts",
        python_callable=_store_alerts,
        doc="Almacena alertas en PostgreSQL y genera reporte",
    )

    collect >> store
