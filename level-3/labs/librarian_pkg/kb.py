# level-2/labs/librarian_pkg/kb.py
"""Carga e indexa la base de conocimiento en ChromaDB."""

import chromadb

RUTA_KB = "data/knowledge_base.md"
RUTA_COLECCION = "data/chroma_db"
NOMBRE_COLECCION = "conocimiento"


def _leer_parrafos(archivo: str) -> tuple[list[str], list[str]]:
    """Divide el archivo en parrafos y asigna a cada uno la fuente (titulo)."""
    with open(archivo, encoding="utf-8") as f:
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

    return parrafos, fuentes


def indexar_base(archivo: str = RUTA_KB) -> int:
    """Indexa la KB en ChromaDB y devuelve cuantos parrafos se guardaron."""
    client = chromadb.PersistentClient(path=RUTA_COLECCION)
    coleccion = client.get_or_create_collection(NOMBRE_COLECCION)

    parrafos, fuentes = _leer_parrafos(archivo)

    ids_existentes = coleccion.get()["ids"]
    if ids_existentes:
        coleccion.delete(ids=ids_existentes)

    coleccion.add(
        documents=parrafos,
        ids=[f"kb-{i}" for i in range(len(parrafos))],
        metadatas=[{"fuente": f} for f in fuentes],
    )
    return len(parrafos)
