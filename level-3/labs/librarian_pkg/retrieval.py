# level-2/labs/librarian_pkg/retrieval.py
"""Recuperacion: busca los parrafos mas relevantes a una pregunta."""

from labs.librarian_pkg.kb import NOMBRE_COLECCION, RUTA_COLECCION
import chromadb


def recuperar(pregunta: str, n: int = 3) -> list[tuple[str, str]]:
    """Devuelve (parrafo, fuente) de los n mas relevantes a la pregunta."""
    client = chromadb.PersistentClient(path=RUTA_COLECCION)
    coleccion = client.get_collection(NOMBRE_COLECCION)

    resultados = coleccion.query(query_texts=[pregunta], n_results=n)
    parrafos = resultados["documents"][0]
    fuentes = [m["fuente"] for m in resultados["metadatas"][0]]
    return list(zip(parrafos, fuentes))
