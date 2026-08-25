# level-1/labs/prompt_advanced.py
"""
Prompt avanzado: few-shot, cadena de pensamiento y salida JSON.

Usage:
    python labs/prompt_advanced.py
"""

import json

import requests

OLLAMA_HOST = "http://localhost:11434"


def chat(messages: list[dict], model: str = "llama3.2", temperature: float = 0.2) -> str:
    """Envia una conversacion completa a Ollama y devuelve la respuesta."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def main() -> None:
    # ══════════════════════════════════════════════
    # 1. Few-shot: dar ejemplos en el prompt
    # ══════════════════════════════════════════════
    few_shot = [
        {
            "role": "system",
            "content": (
                "Clasifica el sentimiento de un texto: POSITIVO, NEGATIVO o NEUTRO. "
                "Responde solo con la palabra."
            ),
        },
        {"role": "user", "content": "Me encanta este curso"},
        {"role": "assistant", "content": "POSITIVO"},
        {"role": "user", "content": "El servicio es lento y malo"},
        {"role": "assistant", "content": "NEGATIVO"},
        {"role": "user", "content": "El clima esta normal hoy"},
        {"role": "assistant", "content": "NEUTRO"},
        {"role": "user", "content": "La pelicula fue increible, la recomiendo"},
    ]
    print("=== 1. FEW-SHOT: clasificacion de sentimiento ===")
    print(chat(few_shot))

    # ══════════════════════════════════════════════
    # 2. Cadena de pensamiento: razonar paso a paso
    # ══════════════════════════════════════════════
    cot = [
        {
            "role": "system",
            "content": (
                "Resuelve problemas paso a paso. "
                "Explica el razonamiento y termina con 'Respuesta: <resultado>'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Una camiseta cuesta 25 euros. Tiene un descuento del 20% "
                "y ademas hay que sumar 3 euros de envio. ¿Cuanto se paga en total?"
            ),
        },
    ]
    print()
    print("=== 2. CADENA DE PENSAMIENTO ===")
    print(chat(cot))

    # ══════════════════════════════════════════════
    # 3. Salida estructurada: JSON
    # ══════════════════════════════════════════════
    json_prompt = [
        {
            "role": "system",
            "content": (
                "Extrae la informacion del texto del usuario y devuelve "
                "SOLO un JSON valido con este formato: "
                '{"nombre": string, "edad": number, "ciudad": string}'
            ),
        },
        {"role": "user", "content": "Me llamo Ana, tengo 29 anos y vivo en Madrid"},
    ]
    print()
    print("=== 3. SALIDA JSON ===")
    respuesta = chat(json_prompt)
    print(respuesta)
    try:
        inicio = respuesta.find("{")
        fin = respuesta.rfind("}") + 1
        datos = json.loads(respuesta[inicio:fin])
        print()
        print("JSON parseado correctamente:")
        print(f"  nombre : {datos['nombre']}")
        print(f"  edad   : {datos['edad']}")
        print(f"  ciudad : {datos['ciudad']}")
    except (json.JSONDecodeError, ValueError):
        print("El modelo no devolvio JSON valido.")


if __name__ == "__main__":
    main()