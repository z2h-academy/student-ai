# packaging_demo/text_utils/cleaning.py
"""
Modulo de limpieza de texto.

Funciones para normalizar y limpiar texto antes de
enviarlo a un modelo LLM o procesarlo.
"""

import re
from typing import List


def normalize_spaces(text: str) -> str:
    """Reemplaza multiples espacios, tabs y newlines por un solo espacio.

    "Hola    mundo\\ncomo  estas" -> "Hola mundo como estas"

    Args:
        text: Texto original con espacios irregulares

    Returns:
        Texto con espacios normalizados
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text: str, keep: str = "") -> str:
    """Elimina caracteres especiales del texto.

    "Hola! como estas?" -> "Hola como estas"

    Args:
        text: Texto original
        keep: Caracteres especiales a conservar (ej: "!?")

    Returns:
        Texto solo con letras, numeros, espacios y caracteres en 'keep'
    """
    pattern = f"[^a-zA-Z0-9\\s{re.escape(keep)}]"
    return re.sub(pattern, "", text)


def to_lowercase(text: str) -> str:
    """Convierte todo el texto a minusculas."""
    return text.lower()


# ═══════════════════════════════════════════════════════════════
# Pipeline de limpieza combinada
# ═══════════════════════════════════════════════════════════════

def clean_pipeline(text: str, lowercase: bool = True, keep: str = "") -> str:
    """Ejecuta todas las limpiezas en orden.

    Este es un "pipeline": una secuencia de transformaciones
    que se aplican una tras otra. Es un patron comun en IA.

    Args:
        text: Texto a limpiar
        lowercase: Si se debe convertir a minusculas
        keep: Caracteres especiales a conservar

    Returns:
        Texto limpio
    """
    text = normalize_spaces(text)
    text = remove_special_chars(text, keep=keep)
    if lowercase:
        text = to_lowercase(text)
    return text
