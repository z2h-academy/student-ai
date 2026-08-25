# level-2/labs/test_api.py
"""Tests de la API librarian con pytest + TestClient de FastAPI."""

import pytest
from fastapi.testclient import TestClient

from labs.api_main import app
from labs import api_main


@pytest.fixture(scope="module")
def client():
    """Cliente de pruebas que apunta a la app (sin levantar uvicorn)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def indexar(client):
    """Indexa la KB una sola vez para toda la bateria de tests."""
    client.post("/api/index")


def test_health(client):
    """El endpoint /health debe reportar que el servicio esta vivo."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_devuelve_parrafos(client):
    """El endpoint /api/index debe devolver la cantidad indexada."""
    resp = client.post("/api/index")
    assert resp.status_code == 200
    assert resp.json()["indexados"] > 0


def test_ask_responde_con_citas(client, monkeypatch):
    """/api/ask debe devolver pregunta, respuesta y fuentes."""

    def fake_generar(pregunta, contexto):
        return f"Respuesta de prueba. (Fuente: {contexto[0][1]})"

    monkeypatch.setattr(api_main, "generar", fake_generar)
    resp = client.post("/api/ask", json={"pregunta": "¿Que es una API REST?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pregunta"] == "¿Que es una API REST?"
    assert "Respuesta de prueba" in data["respuesta"]
    assert len(data["fuentes"]) > 0


def test_ask_pregunta_vacia_devuelve_422(client):
    """Una pregunta vacia debe ser rechazada por Pydantic (422)."""
    resp = client.post("/api/ask", json={"pregunta": ""})
    assert resp.status_code == 422


def test_ask_sin_campo_devuelve_422(client):
    """Un request sin el campo 'pregunta' debe ser rechazado (422)."""
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422
