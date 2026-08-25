# level-1/labs/tokens_demo.py
"""
Tokens y limites de contexto: medir con tiktoken y truncar.

Usage:
    python labs/tokens_demo.py
"""

import tiktoken


def contar_tokens(texto: str, modelo: str = "gpt-4o") -> int:
    """Cuenta los tokens de un texto usando el tokenizador del modelo."""
    enc = tiktoken.encoding_for_model(modelo)
    return len(enc.encode(texto))


def truncar(texto: str, max_tokens: int, modelo: str = "gpt-4o") -> str:
    """Trunca un texto a un maximo de tokens, conservando el principio."""
    enc = tiktoken.encoding_for_model(modelo)
    tokens = enc.encode(texto)
    if len(tokens) <= max_tokens:
        return texto
    return enc.decode(tokens[:max_tokens])


def main() -> None:
    textos = [
        "hello world",
        "Hola mundo",
        "La inteligencia artificial transforma industrias enteras.",
        "supercalifragilisticoespialidoso",
    ]

    print("=== CONTEO DE TOKENS ===")
    for texto in textos:
        print(f"{contar_tokens(texto):>4} tokens | {texto}")

    print()
    print("=== RELACION PALABRAS vs TOKENS (español) ===")
    parrafo = (
        "El aprendizaje automatico es una rama de la inteligencia artificial "
        "que permite a las computadoras aprender de los datos sin ser "
        "programadas explicitamente para cada tarea."
    )
    palabras = len(parrafo.split())
    tokens = contar_tokens(parrafo)
    print(f"palabras: {palabras}")
    print(f"tokens:   {tokens}")
    print(f"relacion: {tokens / palabras:.2f} tokens por palabra")

    print()
    print("=== TRUNCACION ===")
    texto_largo = " ".join(["token de ejemplo numero"] * 100)
    print(f"texto original: {contar_tokens(texto_largo)} tokens")
    cortado = truncar(texto_largo, 50)
    print(f"truncado a 50:  {contar_tokens(cortado)} tokens")
    print(f"final del texto truncado: ...{cortado[-30:]}")


if __name__ == "__main__":
    main()