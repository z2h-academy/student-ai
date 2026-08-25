"""
retriever.py — Búsqueda semántica en ChromaDB

Función retrieve() que dado una query, genera su embedding con
sentence-transformers y busca los k chunks más relevantes en el
índice de ChromaDB.

Retorna lista de dicts con chunk_id, text, source (product_id) y score.
"""

import logging
import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kb_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def _get_embedder() -> SentenceTransformer:
    """Carga el modelo de embeddings (lazy, se cachea en memoria)."""
    if not hasattr(_get_embedder, "_instance"):
        logger.info(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
        _get_embedder._instance = SentenceTransformer(EMBEDDING_MODEL)
    return _get_embedder._instance


def _get_collection(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.Collection:
    """Obtiene la colección de ChromaDB."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    query: str,
    k: int = 5,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> list[dict]:
    """
    Busca los k chunks más relevantes para una query.

    Args:
        query: Pregunta o texto de búsqueda.
        k: Número de resultados a retornar.
        persist_dir: Directorio de persistencia de ChromaDB.

    Returns:
        Lista de dicts ordenados por relevancia (score descendente):
        [
            {
                "chunk_id": str,     # ID del chunk en ChromaDB
                "text": str,         # Texto del chunk
                "source": str,       # product_id de origen
                "score": float,      # Score de similitud (0-1, mayor = más relevante)
            },
            ...
        ]
    """
    if not query or not query.strip():
        return []

    embedder = _get_embedder()
    collection = _get_collection(persist_dir)

    # Verificar que la colección tiene datos
    if collection.count() == 0:
        logger.warning("Colección vacía — ¿se ejecutó index_kb.py?")
        return []

    # Generar embedding de la query
    query_embedding = embedder.encode([query])[0].tolist()

    # Buscar en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Desempaquetar resultados
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []
    ids = results["ids"][0] if results["ids"] else []

    # Construir lista de resultados
    retrieved: list[dict] = []
    for idx in range(len(documents)):
        # ChromaDB con cosine distance retorna distancia = 1 - cosine_similarity
        # Convertir a score de similitud (0-1)
        score = 1.0 - distances[idx] if idx < len(distances) else 0.0

        metadata = metadatas[idx] if idx < len(metadatas) else {}
        retrieved.append({
            "chunk_id": ids[idx] if idx < len(ids) else f"unknown_{idx}",
            "text": documents[idx],
            "source": metadata.get("product_id", "unknown"),
            "score": round(score, 4),
        })

    logger.info(
        f"Retrieve: query='{query[:50]}...' → {len(retrieved)} resultados "
        f"(top score: {retrieved[0]['score']:.4f})" if retrieved else
        f"Retrieve: query='{query[:50]}...' → 0 resultados"
    )

    return retrieved
