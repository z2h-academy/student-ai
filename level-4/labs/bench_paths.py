# level-4/labs/bench_paths.py
"""
Benchmark de rutas: base vs prompt engineering vs LoRA vs QLoRA.

Las 4 rutas usan el MISMO modelo base (SmolLM2-360M): lo que cambia es
la tecnica. Mide latencia y calidad (heurstica de palabras clave) y
termina con la decision: cuando justifica fine-tunear.

Usage:
    python labs/bench_paths.py
"""

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO_BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
RUTA_LORA = Path(__file__).resolve().parent.parent / "adapters" / "lora"
RUTA_QLORA = Path(__file__).resolve().parent.parent / "adapters" / "qlora"

SYSTEM = (
    "Eres 'librarian', un asistente tecnico que responde en espanol "
    "de forma breve y clara."
)

# Preguntas de evaluacion + palabras clave que una buena respuesta toca
EVALS = [
    {
        "pregunta": "Que es una API REST?",
        "claves": ["http", "api", "rest"],
    },
    {
        "pregunta": "Como funciona el RAG?",
        "claves": ["recupera", "contexto", "busqu"],
    },
    {
        "pregunta": "Para que sirve ChromaDB?",
        "claves": ["vector", "embedding", "base"],
    },
]


def responder(modelo, tokenizer, pregunta: str, system: str | None,
              max_tokens: int = 80) -> tuple[str, float]:
    """Genera respuesta y devuelve (texto, latencia_s)."""
    mensajes = ([{"role": "system", "content": system}] if system else [])
    mensajes.append({"role": "user", "content": pregunta})
    prompt = tokenizer.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entrada = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    inicio = time.perf_counter()
    with torch.no_grad():
        salida = modelo.generate(
            entrada["input_ids"], max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latencia = time.perf_counter() - inicio
    texto = tokenizer.decode(
        salida[0][entrada["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return texto.strip(), round(latencia, 2)


def calidad(texto: str, claves: list[str]) -> int:
    """Cuenta cuantas palabras clave esperadas aparecen en la respuesta."""
    bajo = texto.lower()
    return sum(1 for c in claves if c in bajo)


def cargar_rutas(tokenizer, dispositivo: str) -> dict:
    """Carga las 4 variantes del mismo modelo base."""
    print("Cargando modelo base (x3 copias para comparar)...")
    modelos = {}
    modelos["base"] = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, torch_dtype=torch.float32
    ).to(dispositivo).eval()

    ft = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE, torch_dtype=torch.float32
    ).to(dispositivo)
    modelos["prompt"] = ft.eval()

    if RUTA_LORA.exists():
        copia_lora = AutoModelForCausalLM.from_pretrained(
            MODELO_BASE, torch_dtype=torch.float32
        ).to(dispositivo)
        modelos["lora"] = PeftModel.from_pretrained(copia_lora, str(RUTA_LORA)).eval()

    if RUTA_QLORA.exists() and torch.cuda.is_available():
        copia_q = AutoModelForCausalLM.from_pretrained(
            MODELO_BASE, torch_dtype=torch.bfloat16
        ).to(dispositivo)
        modelos["qlora"] = PeftModel.from_pretrained(copia_q, str(RUTA_QLORA)).eval()

    return modelos


def main() -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== BENCH DE RUTAS (mismo modelo base, distinta tecnica) ===")
    print(f"Modelo: {MODELO_BASE} | Dispositivo: {dispositivo}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
    rutas = cargar_rutas(tokenizer, dispositivo)

    # Config por ruta: (usa_system,)
    configs = {"base": False, "prompt": True, "lora": True, "qlora": True}

    resultados: dict[str, dict] = {}
    for nombre, modelo in list(rutas.items()):
        usa_system = configs[nombre]
        latencias, calidades = [], []
        for ev in EVALS:
            texto, latencia = responder(
                modelo, tokenizer, ev["pregunta"],
                SYSTEM if usa_system else None,
            )
            latencias.append(latencia)
            calidades.append(calidad(texto.lower(), ev["claves"]))
        resultados[nombre] = {
            "latencia_promedio": round(sum(latencias) / len(latencias), 2),
            "calidad_total": sum(calidades),
            "calidad_maxima": len(EVALS) * max(len(ev["claves"]) for ev in EVALS),
        }
        del rutas[nombre]
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'ruta':<8} {'latencia prom':>14} {'calidad':>18}")
    print("-" * 44)
    for nombre, r in resultados.items():
        print(
            f"{nombre:<8} {r['latencia_promedio']:>11.2f}s "
            f"{r['calidad_total']:>10}/{r['calidad_maxima']}"
        )

    tamanio_lora = sum(
        f.stat().st_size for f in RUTA_LORA.rglob("*") if f.is_file()
    ) / 1024 / 1024 if RUTA_LORA.exists() else 0

    print("\n--- DECISION DOCUMENTADA ---")
    print(f"Costo del adaptador LoRA: {tamanio_lora:.1f} MB (~0.5% del modelo)")
    print("Reglas practicas:")
    print("  1. Prompt engineering primero: gratis e instantaneo; si alcanza, no FT.")
    print("  2. FT solo con datos suficientes y repetitivos (tono, formato, dominio).")
    print("  3. Con pocos datos, el FT memoriza (overfitting): validar siempre.")
    print("  4. QLoRA cuando el modelo no entra en la GPU disponible.")
    print("  5. Medir antes y despues: sin benchmark, el FT es fe.")


if __name__ == "__main__":
    main()