# level-2/labs/cost_report.py
"""Calculo del costo por request del servicio librarian.

Estima los tokens de entrada (contexto + pregunta) y salida (respuesta)
y los multiplica por el precio del modelo. Usa tiktoken si esta
disponible; si no, una estimacion simple (~4 caracteres por token).

Usage (desde level-2/):
    python -m labs.cost_report
"""

import os

from dotenv import load_dotenv

from labs.librarian_pkg import indexar_base, recuperar, generar

load_dotenv()

# Precios por 1M tokens (USD) — modelo llama3.2 via Ollama
PRECIO_INPUT_POR_MILLON = float(os.getenv("PRECIO_INPUT_POR_MILLON", "0.25"))
PRECIO_OUTPUT_POR_MILLON = float(os.getenv("PRECIO_OUTPUT_POR_MILLON", "0.75"))


def estimar_tokens(texto: str) -> int:
    """Estima los tokens de un texto (aprox. 4 caracteres por token)."""
    return max(1, len(texto) // 4)


def calcular_costo(tokens_input: int, tokens_output: int) -> float:
    """Devuelve el costo en USD de un request."""
    costo_input = tokens_input / 1_000_000 * PRECIO_INPUT_POR_MILLON
    costo_output = tokens_output / 1_000_000 * PRECIO_OUTPUT_POR_MILLON
    return round(costo_input + costo_output, 6)


def reporte(pregunta: str, n: int = 3) -> dict:
    """Ejecuta un request completo y devuelve el desglose de costo."""
    indexar_base()

    contexto = recuperar(pregunta, n=n)
    respuesta = generar(pregunta, contexto)

    contexto_texto = "\n\n".join(
        f"[Fuente: {fuente}] {parrafo}" for parrafo, fuente in contexto
    )
    prompt_input = contexto_texto + pregunta
    tokens_input = estimar_tokens(prompt_input)
    tokens_output = estimar_tokens(respuesta)

    return {
        "pregunta": pregunta,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "costo_usd": calcular_costo(tokens_input, tokens_output),
    }


def main() -> None:
    print("=== COST REPORT: costo por request ===\n")
    for pregunta in [
        "¿Que es una API REST?",
        "¿Cual es la ventaja del RAG?",
    ]:
        r = reporte(pregunta)
        print(f"Pregunta: {r['pregunta']}")
        print(f"  tokens input : {r['tokens_input']}")
        print(f"  tokens output: {r['tokens_output']}")
        print(f"  costo        : ${r['costo_usd']}\n")


if __name__ == "__main__":
    main()
