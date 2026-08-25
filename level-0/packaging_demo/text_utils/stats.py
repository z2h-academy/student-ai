# packaging_demo/text_utils/stats.py
"""
Modulo de estadisticas de texto.

Funciones para analizar metricas basicas de texto:
conteo de palabras, longitud promedio, palabras mas frecuentes.
"""

from collections import Counter
from typing import List, Tuple


def word_count(text: str) -> int:
    """Cuenta la cantidad de palabras en un texto.

    Usa split() que separa por cualquier espacio en blanco.

    Args:
        text: Texto a analizar

    Returns:
        Numero de palabras
    """
    return len(text.split())


def avg_word_length(text: str) -> float:
    """Calcula la longitud promedio de las palabras.

    Args:
        text: Texto a analizar

    Returns:
        Longitud promedio en caracteres
    """
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def most_common_words(text: str, n: int = 5) -> List[Tuple[str, int]]:
    """Devuelve las N palabras mas frecuentes.

    Usa Counter que es una subclase de dict especializada
    en conteo de elementos.

    Args:
        text: Texto a analizar
        n: Cantidad de palabras a devolver

    Returns:
        Lista de tuplas (palabra, frecuencia)
    """
    words = text.lower().split()
    return Counter(words).most_common(n)
