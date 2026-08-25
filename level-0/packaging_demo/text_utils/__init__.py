# packaging_demo/text_utils/__init__.py
"""
text_utils - Utilidades para procesamiento de texto.

Este paquete contiene funciones para limpiar, normalizar
y analizar texto. Compatible con datos de entrada para LLMs.
"""

from .cleaning import normalize_spaces, remove_special_chars
from .stats import word_count, avg_word_length, most_common_words

__version__ = "0.1.0"
