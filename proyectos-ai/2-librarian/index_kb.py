"""
index_kb.py — Indexa la Knowledge Base en ChromaDB

Carga los chunks y embeddings pre-computados (outputs del Proyecto 1)
y los indexa en ChromaDB para búsqueda semántica.

Inputs:
  - silver/kb/chunks.parquet  → product_id, chunk
  - gold/embeddings/kb_chunks.parquet → product_id, chunk, embedding

Output:
  - ./chroma_db/ → directorio de persistencia de ChromaDB
"""

import json
import logging
import os
import pathlib
import sys

import chromadb
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kb_chunks"

load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def _resolve_parquet(path: str) -> pathlib.Path:
    """Resuelve una ruta de Parquet soportando MinIO (s3://) y local."""
    p = pathlib.Path(path)
    if p.exists():
        return p
    raise FileNotFoundError(f"Parquet no encontrado: {path}")


def load_chunks(chunks_path: str) -> pd.DataFrame:
    """Lee chunks.parquet (product_id, chunk)."""
    resolved = _resolve_parquet(chunks_path)
    df = pd.read_parquet(resolved)
    required = {"product_id", "chunk"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"chunks.parquet falta columnas: {missing}")
    logger.info(f"Chunks cargados: {len(df):,} filas desde {resolved}")
    return df


def load_embeddings(embeddings_path: str) -> pd.DataFrame:
    """Lee kb_chunks.parquet (product_id, chunk, embedding)."""
    resolved = _resolve_parquet(embeddings_path)
    df = pd.read_parquet(resolved)
    required = {"product_id", "chunk", "embedding"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"kb_chunks.parquet falta columnas: {missing}")
    logger.info(f"Embeddings cargados: {len(df):,} filas desde {resolved}")
    return df


def build_index(
    df_embeddings: pd.DataFrame,
    persist_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Indexa chunks + embeddings en ChromaDB.

    Genera chunk_ids deterministas basados en product_id + índice
    para evitar duplicados entre re-indexaciones.

    Returns:
        Colección de ChromaDB lista para consultas.
    """
    client = chromadb.PersistentClient(path=persist_dir)

    # Upsert: si la colección ya existe, la borramos y re-creamos
    try:
        client.delete_collection(collection_name)
        logger.info(f"Colección '{collection_name}' eliminada para re-indexación")
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Generar IDs deterministas
    chunk_ids: list[str] = []
    for idx in range(len(df_embeddings)):
        row = df_embeddings.iloc[idx]
        chunk_ids.append(f"{row['product_id']}_{idx}")

    # Extraer datos
    documents = df_embeddings["chunk"].tolist()
    embeddings = df_embeddings["embedding"].tolist()
    metadatas = [
        {"product_id": str(row["product_id"])}
        for row in df_embeddings.to_dict("records")
    ]

    # Indexar en lotes de 5000 (límite de ChromaDB por batch)
    batch_size = 5000
    total = len(chunk_ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=chunk_ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
        logger.info(f"Indexados {end:,}/{total:,} chunks")

    logger.info(f"Índice creado: {total:,} chunks en '{collection_name}'")
    return collection


def run(
    chunks_path: str,
    embeddings_path: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> dict:
    """
    Flujo completo: carga parquets → indexa en ChromaDB.

    Returns:
        dict con métricas del proceso.
    """
    df_chunks = load_chunks(chunks_path)
    df_embeddings = load_embeddings(embeddings_path)

    # Validar que los chunks coinciden
    if len(df_chunks) != len(df_embeddings):
        logger.warning(
            f"Mismatch chunks ({len(df_chunks):,}) vs embeddings "
            f"({len(df_embeddings):,}). Usando embeddings como fuente."
        )

    collection = build_index(df_embeddings, persist_dir=persist_dir)

    metrics = {
        "chunks_indexed": collection.count(),
        "persist_dir": persist_dir,
        "collection": COLLECTION_NAME,
    }
    logger.info(f"Indexación completa: {json.dumps(metrics)}")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Indexa la KB en ChromaDB — Proyecto 2 (librarian)"
    )
    parser.add_argument(
        "--chunks",
        default="silver/kb/chunks.parquet",
        help="Ruta a chunks.parquet (default: silver/kb/chunks.parquet)",
    )
    parser.add_argument(
        "--embeddings",
        default="gold/embeddings/kb_chunks.parquet",
        help="Ruta a kb_chunks.parquet (default: gold/embeddings/kb_chunks.parquet)",
    )
    parser.add_argument(
        "--persist-dir",
        default=CHROMA_PERSIST_DIR,
        help=f"Directorio de ChromaDB (default: {CHROMA_PERSIST_DIR})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        metrics = run(args.chunks, args.embeddings, args.persist_dir)
        print("=" * 60)
        print(
            f"INDEXACIÓN COMPLETA | "
            f"chunks={metrics['chunks_indexed']:,} | "
            f"dir={metrics['persist_dir']}"
        )
        print("=" * 60)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fallo en index_kb: {e}", exc_info=True)
        sys.exit(1)
