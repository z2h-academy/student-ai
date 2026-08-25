# level-4/labs/serve_finetuned.py
"""
Servir el modelo fine-tuneado: compara el modelo BASE contra el FT.

Carga SmolLM2 dos veces (limpio y con el adaptador LoRA de Level 3)
y hace la misma pregunta a ambos: la diferencia muestra que el FT
aprendio el dominio. Al final genera el Modelfile para montar el
adaptador en Ollama (el servidor del nivel).

Usage:
    python labs/serve_finetuned.py
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO_BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
RUTA_ADAPTER = Path(__file__).resolve().parent.parent / "adapters" / "lora"
RUTA_MODEFILE = Path(__file__).resolve().parent / "Modelfile"

PREGUNTAS = [
    "Que es una API REST?",
    "Como funciona el RAG?",
]


def responder(modelo, tokenizer, pregunta: str, max_tokens: int = 80) -> str:
    """Genera una respuesta con chat template."""
    mensajes = [
        {
            "role": "system",
            "content": (
                "Eres 'librarian', un asistente tecnico que responde "
                "en espanol de forma breve y clara."
            ),
        },
        {"role": "user", "content": pregunta},
    ]
    prompt = tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entrada = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    with torch.no_grad():
        salida = modelo.generate(
            entrada["input_ids"], max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    texto = tokenizer.decode(
        salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return texto.strip()


def crear_modelfile() -> None:
    """Genera el Modelfile para montar el adaptador LoRA en Ollama."""
    contenido = f"""# Modelfile: librarian fine-tuneado con LoRA
FROM smollm2:360m
ADAPTER /tmp/lora.gguf

SYSTEM \"\"\"Eres 'librarian', un asistente tecnico especializado en APIs,
RAG y sistemas de IA. Responde en espanol, breve y con fuentes.\"\"\"

PARAMETER temperature 0.3
"""
    RUTA_MODEFILE.write_text(contenido, encoding="utf-8")
    print(f"Modelfile generado en {RUTA_MODEFILE}")
    print("\nPara montarlo en Ollama (el adaptador debe ser GGUF y estar")
    print("accesible para el contenedor de Ollama):\n")
    print("  # 1. Convertir el adaptador PEFT a GGUF (llama.cpp):")
    print("  python convert_lora_to_gguf.py adapters/lora \\")
    print("      --outfile adapters/lora.gguf --outtype f16 \\")
    print(f"      --base-model-id {MODELO_BASE}")
    print("  # 2. Copiarlo al contenedor junto al Modelfile:")
    print("  docker cp adapters/lora.gguf ollama:/tmp/lora.gguf")
    print(f"  docker cp {RUTA_MODEFILE} ollama:/tmp/Modelfile")
    print("  # 3. Crear y probar el modelo:")
    print("  docker exec ollama ollama create mentor -f /tmp/Modelfile")
    print("  curl http://localhost:11434/api/chat -d '{...\"model\":\"mentor\"...}'")


def main() -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== SERVIR EL MODELO FINE-TUNEADO ===")
    print(f"Dispositivo: {dispositivo}")

    print("\n1. Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    print("2. Cargando modelo BASE...")
    # Instancia separada de la del FT: PEFT inyecta el adaptador EN el
    # modelo que recibe, asi que comparar requiere dos copias.
    base = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32)
    base.to(dispositivo).eval()

    print("3. Cargando modelo FINE-TUNEADO (copia + adaptador)...")
    ft_copia = AutoModelForCausalLM.from_pretrained(MODELO_BASE, torch_dtype=torch.float32)
    ft_copia.to(dispositivo)
    ft = PeftModel.from_pretrained(ft_copia, str(RUTA_ADAPTER))
    ft.eval()

    print("\n--- COMPARATIVA ---")
    for pregunta in PREGUNTAS:
        print(f"\nPregunta: {pregunta}")
        respuesta_base = responder(base, tokenizer, pregunta)
        respuesta_ft = responder(ft, tokenizer, pregunta)
        print(f"BASE> {respuesta_base[:180]}")
        print(f"FT  > {respuesta_ft[:180]}")

    del base, ft, ft_copia
    if dispositivo == "cuda":
        torch.cuda.empty_cache()

    print()
    crear_modelfile()


if __name__ == "__main__":
    main()