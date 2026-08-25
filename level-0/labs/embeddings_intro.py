# level-0/labs/embeddings_intro.py
"""
Demonstracion de embeddings y similitud semantica.

Usa sentence-transformers para convertir texto en vectores
y calcula similitud coseno entre ellos.

Usage:
    python labs/embeddings_intro.py
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. Cargar el modelo de embeddings
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("DEMO: Embeddings y Similitud Semantica")
    print("=" * 60)

    # Cargamos un modelo liviano de sentence-transformers
    # all-MiniLM-L6-v2: 384 dimensiones, ~80MB, rapido
    # Es uno de los mejores modelos relacion calidad/velocidad
    print("\n1. Cargando modelo de embeddings...")
    print("   (primera vez descarga ~80MB, siguiente usa cache)")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    dimension = modelo.get_embedding_dimension()
    print(f"   Modelo: {modelo}")
    print(f"   Dimension del embedding: {dimension}")

    # ═══════════════════════════════════════════════════════════
    # 2. Generar embeddings
    # ═══════════════════════════════════════════════════════════

    print("\n2. Generando embeddings...")
    frases = [
        "Los gatos son mascotas fantasticas",
        "Los perros son animales leales",
        "Me gusta programar en Python",
        "El lenguaje Python es muy versatil",
        "El clima hoy esta soleado",
    ]

    # encode() convierte CADA frase en un vector de 384 numeros
    embeddings = modelo.encode(frases)
    print(f"   Frases: {len(frases)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   (filas=frases, columnas=dimension)")

    # Mostrar los primeros valores del embedding de la frase 0
    print(f"\n   Embedding de '{frases[0]}':")
    print(f"   Primeros 8 valores: {embeddings[0][:8]}")
    print(f"   ... total {len(embeddings[0])} valores")

    # ═══════════════════════════════════════════════════════════
    # 3. Calcular similitud coseno
    # ═══════════════════════════════════════════════════════════

    print("\n3. Matriz de similitud coseno:")
    # cosine_similarity compara CADA frase con TODAS las demas
    # Resultado: matriz 5x5 donde [i][j] = similitud entre frase i y j
    matriz_similitud = cosine_similarity(embeddings)

    # Mostrar la matriz formateada
    # Los nombres cortos para que entre en la pantalla
    nombres = ["Gatos", "Perros", "Python1", "Python2", "Clima"]
    print(f"\n   {'':>8}", end="")
    for n in nombres:
        print(f"{n:>8}", end="")
    print()
    for i, n in enumerate(nombres):
        print(f"   {n:>8}", end="")
        for j in range(len(nombres)):
            print(f"{matriz_similitud[i][j]:>8.3f}", end="")
        print()

    # ═══════════════════════════════════════════════════════════
    # 4. Interpretar los resultados
    # ═══════════════════════════════════════════════════════════

    print("\n4. Interpretacion:")
    # Las frases MAS similares
    for i in range(len(frases)):
        for j in range(i + 1, len(frases)):
            sim = matriz_similitud[i][j]
            nivel = (
                "MUY similar"
                if sim > 0.7
                else "similar"
                if sim > 0.5
                else "poco similar"
                if sim > 0.3
                else "diferente"
            )
            print(f"   {nombres[i]:>8} vs {nombres[j]:<8}: {sim:.3f} ({nivel})")

    # ═══════════════════════════════════════════════════════════
    # 5. Busqueda semantica (el caso de uso real)
    # ═══════════════════════════════════════════════════════════

    print("\n5. Busqueda semantica:")
    # Dada una consulta, encontrar la frase mas similar
    consulta = "Los animales domesticos"
    embedding_consulta = modelo.encode([consulta])
    similitudes = cosine_similarity(embedding_consulta, embeddings)[0]
    indice_mas_similar = np.argmax(similitudes)

    print(f"   Consulta: '{consulta}'")
    print(f"   Resultado: '{frases[indice_mas_similar]}'")
    print(f"   Similitud: {similitudes[indice_mas_similar]:.3f}")

    print("\n   Todas las similitudes:")
    for i, frase in enumerate(frases):
        print(f"   {similitudes[i]:.3f} - {frase}")

    # ═══════════════════════════════════════════════════════════
    # 6. Embeddings NO son bolsas de palabras
    # ═══════════════════════════════════════════════════════════

    print("\n6. Embeddings capturan significado, no solo palabras:")
    # Dos formas de decir lo mismo
    frases_similar = [
        "Me encanta la inteligencia artificial",
        "La IA me fascina profundamente",
        "Hoy desayune un cafe con leche",
    ]
    emb_similar = modelo.encode(frases_similar)
    sim_ia_cafe = cosine_similarity(emb_similar)

    print(f"   'Me encanta la IA' vs 'La IA me fascina' = {sim_ia_cafe[0][1]:.3f}")
    print(f"   'Me encanta la IA' vs 'cafe con leche'   = {sim_ia_cafe[0][2]:.3f}")
    # Las dos frases sobre IA deberian tener similitud ALTA (>0.7)
    # La frase del cafe deberia tener similitud BAJA con IA (<0.3)

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
