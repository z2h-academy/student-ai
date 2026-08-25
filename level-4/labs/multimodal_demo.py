# level-4/labs/multimodal_demo.py
"""
Pipeline multimodal: texto + imagen con un modelo de vision (qwen2.5vl).

Genera una imagen local (con Pillow), la envia al modelo multimodal
de Ollama en base64 y pide que describa lo que ve.

Usage:
    python labs/multimodal_demo.py
"""

import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODELO = "qwen2.5vl:3b"
RUTA_IMAGEN = "data/imagen_test.png"


def crear_imagen(ruta: str) -> None:
    """Genera una imagen simple y reconocible: un semaforo dibujado."""
    img = Image.new("RGB", (300, 500), "darkgray")
    draw = ImageDraw.Draw(img)

    # Caja del semaforo
    draw.rectangle([100, 40, 200, 340], fill="black", outline="white", width=3)
    # Luces: rojo encendido, amarillo y verde apagados
    draw.ellipse([120, 60, 180, 120], fill="red")
    draw.ellipse([120, 140, 180, 200], fill="dimgray")
    draw.ellipse([120, 220, 180, 280], fill="dimgray")
    # Poste
    draw.rectangle([140, 340, 160, 480], fill="black")

    img.save(ruta)
    print(f"Imagen generada: {ruta} ({img.size[0]}x{img.size[1]})")


def preguntar_con_imagen(pregunta: str, ruta: str) -> str:
    """Envia texto + imagen al modelo multimodal via /api/chat."""
    imagen_b64 = base64.b64encode(Path(ruta).read_bytes()).decode()
    payload = {
        "model": MODELO,
        "messages": [
            {
                "role": "user",
                "content": pregunta,
                "images": [imagen_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def main() -> None:
    print("=== PIPELINE MULTIMODAL (texto + imagen) ===\n")

    crear_imagen(RUTA_IMAGEN)
    print()

    preguntas = [
        "Que objeto aparece en esta imagen? Describe su estado.",
        "De que color es la luz encendida?",
    ]

    for pregunta in preguntas:
        print(f"PREGUNTA> {pregunta}")
        respuesta = preguntar_con_imagen(pregunta, RUTA_IMAGEN)
        print(f"MODELO > {respuesta[:250]}")
        print()


if __name__ == "__main__":
    main()