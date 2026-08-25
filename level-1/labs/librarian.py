# level-1/labs/librarian.py
"""
Proyecto integrador: librarian — asistente RAG con citas.

Responde preguntas usando la base de conocimiento y cita
la fuente (titulo de la seccion) de cada dato.

Usage:
    python labs/librarian.py
"""

import chromadb
import requests

OLLAMA_HOST = "http://localhost:11434"
MODELO_CHAT = "llama3.2"
RUTA_KB = "data/knowledge_base.md"


def indexar_base() -> None:
    """Indexa la KB guardando el titulo de cada seccion como metadata."""
    client = chromadb.PersistentClient(path="data/chroma_db")
    coleccion = client.get_or_create_collection("conocimiento")

    with open(RUTA_KB, encoding="utf-8") as f:
        contenido = f.read()

    bloques = contenido.split("\n\n")
    parrafos = []
    fuentes = []
    titulo_actual = "Introduccion"

    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue
        if bloque.startswith("## "):
            titulo_actual = bloque.replace("## ", "").strip()
            continue
        if bloque.startswith("# "):
            continue
        parrafos.append(bloque)
        fuentes.append(titulo_actual)

    if coleccion.count() == 0:
        coleccion.add(
            documents=parrafos,
            ids=[f"kb-{i}" for i in range(len(parrafos))],
            metadatas=[{"fuente": f} for f in fuentes],
        )
        print(f"Indexados {len(parrafos)} parrafos con sus fuentes.")


def recuperar(pregunta: str, n: int = 3) -> list[tuple[str, str]]:
    """Devuelve (parrafo, fuente) de los mas relevantes."""
    client = chromadb.PersistentClient(path="data/chroma_db")
    coleccion = client.get_collection("conocimiento")
    resultados = coleccion.query(query_texts=[pregunta], n_results=n)
    parrafos = resultados["documents"][0]
    fuentes = [m["fuente"] for m in resultados["metadatas"][0]]
    return list(zip(parrafos, fuentes))


def generar(pregunta: str, contexto: list[tuple[str, str]]) -> str:
    """Genera la respuesta con citas de fuente."""
    lineas = [f"[Fuente: {fuente}] {parrafo}" for parrafo, fuente in contexto]
    contexto_texto = "\n\n".join(lineas)
    system = (
        "Eres 'librarian', un asistente que responde usando SOLO el contexto. "
        "Al final de cada dato citado, indica la fuente entre parentesis "
        "usando el formato (Fuente: NOMBRE). "
        "Si el contexto no tiene la respuesta, di 'No tengo informacion sobre eso'."
    )
    user = f"Contexto:\n{contexto_texto}\n\nPregunta: {pregunta}"

    payload = {
        "model": MODELO_CHAT,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> None:
    print("=== LIBRARIAN: asistente RAG con citas ===\n")

    indexar_base()

    preguntas = [
        "¿Que es una API REST?",
        "¿Cuando fue fundada Z2H Academy?",
        "¿Cual es la ventaja del RAG?",
    ]

    for pregunta in preguntas:
        print(f"\n--- PREGUNTA: {pregunta} ---")
        contexto = recuperar(pregunta)
        print("Contexto recuperado:")
        for parrafo, fuente in contexto:
            print(f"  * [{fuente}] {parrafo[:70]}...")
        respuesta = generar(pregunta, contexto)
        print(f"Respuesta: {respuesta}")


if __name__ == "__main__":
    main()