# level-1/labs/rag_pipeline.py
"""
RAG completo: indexar base de conocimiento, recuperar, generar.

Usage:
    python labs/rag_pipeline.py
"""

import chromadb
import requests

OLLAMA_HOST = "http://localhost:11434"
MODELO_CHAT = "llama3.2"


def indexar_base(archivo: str) -> None:
    """Lee un archivo de texto plano y lo indexa en ChromaDB."""
    client = chromadb.PersistentClient(path="data/chroma_db")
    coleccion = client.get_or_create_collection("conocimiento")

    with open(archivo, encoding="utf-8") as f:
        contenido = f.read()

    parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]

    if coleccion.count() == 0:
        coleccion.add(documents=parrafos, ids=[f"kb-{i}" for i in range(len(parrafos))])
        print(f"Indexados {len(parrafos)} parrafos de {archivo}")
    else:
        print(f"Coleccion ya tiene {coleccion.count()} parrafos indexados.")


def recuperar(pregunta: str, n: int = 3) -> list[str]:
    """Busca los parrafos mas relevantes a la pregunta."""
    client = chromadb.PersistentClient(path="data/chroma_db")
    coleccion = client.get_collection("conocimiento")
    resultados = coleccion.query(query_texts=[pregunta], n_results=n)
    return resultados["documents"][0]


def generar(pregunta: str, contexto: list[str]) -> str:
    """Genera una respuesta con el contexto recuperado."""
    contexto_texto = "\n\n".join(contexto)
    system = (
        "Eres un asistente que responde SOLO con la informacion del contexto. "
        "Si el contexto no tiene la respuesta, di 'No tengo informacion sobre eso'. "
        "Responde en espanol."
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
    print("=== RAG: RETRIEVAL AUGMENTED GENERATION ===\n")

    print("--- 1. INDEXAR ---")
    indexar_base("data/knowledge_base.md")

    preguntas = [
        "¿Que es una API REST?",
        "¿Cual es la ventaja del RAG?",
        "¿Donde tiene su sede Z2H Academy?",
        "¿Quien fundo Z2H Academy?",
    ]

    for pregunta in preguntas:
        print(f"\n--- PREGUNTA: {pregunta} ---")
        contexto = recuperar(pregunta)
        print(f"Contexto recuperado ({len(contexto)} parrafos):")
        for c in contexto:
            print(f"  * {c[:80]}...")
        respuesta = generar(pregunta, contexto)
        print(f"Respuesta: {respuesta}")


if __name__ == "__main__":
    main()