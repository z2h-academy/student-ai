# level-1/labs/chroma_demo.py
"""
ChromaDB: almacen vectorial — crear coleccion, insertar, consultar.

Usage:
    python labs/chroma_demo.py
"""

import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")


def main() -> None:
    print("=== CHROMADB: almacen vectorial ===")

    coleccion = client.get_or_create_collection("conocimiento")

    documentos = [
        "El gato duerme en el sofa todas las tardes",
        "Los perros necesitan paseos diarios para mantenerse sanos",
        "La lluvia de la montana riega los bosques de pinos",
        "El clima caluroso aumenta el consumo de agua en la ciudad",
    ]
    ids = ["doc-1", "doc-2", "doc-3", "doc-4"]

    if coleccion.count() == 0:
        coleccion.add(documents=documentos, ids=ids)
        print(f"Insertados {coleccion.count()} documentos.")
    else:
        print(f"La coleccion ya tiene {coleccion.count()} documentos.")

    print()
    print("=== CONSULTA: buscar por similitud ===")
    resultados = coleccion.query(
        query_texts=["¿Donde duerme el gato?"],
        n_results=2,
    )

    for documento, distancia in zip(
        resultados["documents"][0], resultados["distances"][0]
    ):
        print(f"[distancia {distancia:.3f}] {documento}")

    print()
    print("=== CONSULTA 2: tema distinto ===")
    resultados2 = coleccion.query(
        query_texts=["¿Que pasa con el agua cuando hace calor?"],
        n_results=1,
    )
    print(f"[distancia {resultados2['distances'][0][0]:.3f}] "
          f"{resultados2['documents'][0][0]}")


if __name__ == "__main__":
    main()