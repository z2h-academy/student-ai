# packaging_demo/main.py
"""
Punto de entrada que demuestra el uso del paquete text_utils.

Importa funciones desde los modulos del paquete y las ejecuta.

Usage:
    python main.py
"""

from text_utils.cleaning import clean_pipeline, normalize_spaces
from text_utils.stats import avg_word_length, most_common_words, word_count


def main() -> None:
    print("=" * 60)
    print("DEMO: Packaging y Modulos")
    print("=" * 60)

    texto_original = (
        "  HOLA!!  como estas?  espero que   muy bien!!!  "
        "Este es un TEXTO de prueba con  palabras   repetidas.  "
        "palabras palabras palabras palabras"
    )

    print(f"\nTexto original:      '{texto_original[:50]}...'")

    # Usar funciones del paquete
    texto_limpio = clean_pipeline(texto_original, lowercase=True, keep="")
    print(f"Texto limpio:        '{texto_limpio}'")

    texto_sin_lower = clean_pipeline(texto_original, lowercase=False, keep="!?")
    print(f"Texto (con signos):  '{texto_sin_lower}'")

    # Estadisticas
    print(f"\nPalabras totales:    {word_count(texto_limpio)}")
    print(f"Longitud promedio:   {avg_word_length(texto_limpio):.2f} caracteres")

    print(f"\nPalabras mas frecuentes:")
    for palabra, freq in most_common_words(texto_limpio, n=5):
        print(f"   '{palabra}': {freq} veces")

    print("\n" + "=" * 60)
    print("FIN")
    print("=" * 60)


if __name__ == "__main__":
    main()
