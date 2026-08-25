"""Tool MCP: busqueda en ChromaDB (P2)."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def kb_search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Busca los k chunks mas relevantes en ChromaDB.

    Args:
        query: Texto de la consulta.
        k: Numero de resultados a retornar.

    Returns:
        Lista de diccionarios con content, metadata y score.
    """
    try:
        import chromadb

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name="librarian_docs",
            metadata={"hnsw:space": "cosine"},
        )

        results = collection.query(query_texts=[query], n_results=k)

        chunks: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "content": doc,
                "metadata": meta if meta else {},
                "score": 1.0 - dist if dist <= 1.0 else 0.0,
            })

        return chunks

    except Exception:
        return _mock_kb_search(query, k)


def _mock_kb_search(query: str, k: int) -> list[dict[str, Any]]:
    """Retorna resultados simulados para testing sin ChromaDB."""
    return [
        {
            "content": f"Documento simulado de la KB para: {query}",
            "metadata": {"source": "mock", "chunk_id": str(i)},
            "score": 0.9 - (i * 0.1),
        }
        for i in range(min(k, 3))
    ]
